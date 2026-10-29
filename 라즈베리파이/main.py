"""
라즈베리파이(클라이언트) 송신부

역할:
    1 picamera2 로 카메라 캡처
    2 프레임 차분 기반 모션 게이트로 사람 있음 판단
    3 활성 상태일 때만 프레임을 JPEG 로 인코딩해 메인 PC(서버)로 imagezmq 전송
    - python3 main.py --server tcp://<PC_IP>:5555 ->PC 서버로 JPEG 전송

모션 게이트 주의:
    자세 분석 대상은 가만히 앉은 사람이라 움직임이 거의 없다
    그래서 모션은 켜는 트리거로만 쓰고 마지막 움직임 후 PRESENCE_TIMEOUT 초까지는
    사람 있음 상태를 유지해 송신을 끊지 않는다
"""

import argparse
import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

# 설정
RESOLUTION = (640, 360)      # 측면 자세 분석엔 충분
FPS = 5                      # 자세는 정적이라 5fps면 충분 Analyzer의 FPS와 맞출 것
JPEG_QUALITY = 80           

# 모션 게이트
MOTION_PROC_WIDTH = 320      # 모션 계산용 다운스케일 폭 ( Pi 부하 방지 )
DIFF_THRESH = 25             # 픽셀 밝기 차이 임계값
MOTION_RATIO_THRESH = 0.005  # 변한 픽셀 비율이 이 값을 넘으면 움직인 상태
PRESENCE_TIMEOUT = 20.0      # 마지막 움직임 후 이 시간(초)까지 사람 있음 상태 유지

# picamera2 XRGB8888 요청 시 capture_array() 는 4채널(BGRA) 배열을 준다
# 아래 루프에서 BGRA->BGR 로 변환한다 디버깅때 이미지 색이 뒤바뀌어 보이면 변환을 cv2.COLOR_RGBA2BGR 로 바꾸기

DEBUG_DIR = os.path.expanduser("~/posture_debug")


class MotionGate:
    """프레임 차분으로 움직임을 감지하고 PRESENCE_TIMEOUT 으로 존재 상태를 유지"""

    def __init__(self):
        self.prev = None
        self.last_motion_t = 0.0

    def update(self, frame, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = max(1, int(gray.shape[0] * MOTION_PROC_WIDTH / gray.shape[1]))
        small = cv2.resize(gray, (MOTION_PROC_WIDTH, h))                   
        small = cv2.GaussianBlur(small, (21, 21), 0)

        ratio = 0.0
        if self.prev is not None:
            diff = cv2.absdiff(self.prev, small)
            _, th = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
            ratio = float(np.count_nonzero(th)) / th.size
            if ratio > MOTION_RATIO_THRESH:
                self.last_motion_t = now
        self.prev = small

        active = (now - self.last_motion_t) < PRESENCE_TIMEOUT
        return active, ratio


def make_sender(addr, timeout_ms=5000):
    """imagezmq 송신자 생성 서버 무응답 시 무한 대기하지 않도록 타임아웃 설정"""
    import imagezmq
    import zmq
    s = imagezmq.ImageSender(connect_to=addr)
    s.zmq_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.zmq_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    s.zmq_socket.setsockopt(zmq.LINGER, 0)
    return s


def save_debug(frame, active, ratio, idx):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    img = frame.copy()
    label = f"{'ACTIVE' if active else 'idle'}  motion={ratio*100:.3f}%"
    color = (0, 0, 255) if active else (160, 160, 160)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    path = os.path.join(DEBUG_DIR, f"frame_{idx:05d}.jpg")
    cv2.imwrite(path, img)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", help="PC 서버 주소 주소 : tcp://192.168.0.2:5555")
    # 사용할 PC에 새로운 인바운드 규칙 추가로 5555 해줘야함
    ap.add_argument("--test", action="store_true",
                    help="네트워크 없이 카메라+모션게이트만 검증 (디버그 프레임 저장)")
    ap.add_argument("--name", default="pi-posture", help="송신자 식별 이름")
    args = ap.parse_args()

    if not args.test and not args.server:
        ap.error("--server (주소) or --test 모드를 사용하세요")

    sender = None
    if not args.test:
        sender = make_sender(args.server)
        print(f"[실행] 서버로 전송: {args.server}")
    else:
        print(f"[테스트] 네트워크 없음 디버그 프레임 -> {DEBUG_DIR}")

    # 카메라 처리
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": RESOLUTION},
    )   
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1.0)  # 센서 워밍업

    gate = MotionGate()
    interval = 1.0 / FPS
    dbg_idx = 0
    last_dbg_t = 0.0
    prev_active = None

    # FPS 측정용
    fps_cnt, fps_t = 0, time.time()

    print("시작 (Ctrl-C 로 종료)")
    try:
        while True:
            t0 = time.time()
            frame = picam2.capture_array()
            # XRGB8888 -> 4채널(BGRA) OpenCV용 3채널 BGR 로 변환(알파 제거)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            active, ratio = gate.update(frame, t0)

            if active:
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    if args.test:
                        # 상태 전환 시 또는 2초마다 디버그 프레임 저장
                        if active != prev_active or (t0 - last_dbg_t) > 2.0:
                            p = save_debug(frame, active, ratio, dbg_idx)
                            dbg_idx += 1
                            last_dbg_t = t0
                    else:
                        try:
                            sender.send_jpg(args.name, buf)
                        except Exception as e:
                            print(f"[send 실패] 재연결 시도: {e}")
                            try:
                                sender.close()
                            except Exception:
                                pass
                            time.sleep(2.0)
                            sender = make_sender(args.server)

            # 상태 전환 로그
            if active != prev_active:
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"{'-> ACTIVE (전송 시작)' if active else '-> idle (대기)'}")
                prev_active = active

            # 약 1초마다 실측 FPS / 모션 비율 출력
            fps_cnt += 1
            if t0 - fps_t >= 1.0:
                print(f"  fps≈{fps_cnt}  motion={ratio*100:.3f}%  "
                      f"{'ACTIVE' if active else 'idle'}")
                fps_cnt, fps_t = 0, t0

            # 목표 FPS 로 페이싱
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        picam2.stop()
        if sender is not None:
            sender.close()


if __name__ == "__main__":
    main()
