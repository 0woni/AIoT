"""
흐름: 프레임 입력 -> YOLO pose 추론 -> Analyzer 판정 -> Flask 웹 표시
입력 소스 (--source)
    folder:<경로>  : 폴더의 이미지들을 반복 재생
                   예) --source folder:./debug_frames   (Pi에서 scp로 받은 프레임)
    zmq            : 실제 Pi 에서 imagezmq 로 수신


    python3 pc_server.py --source zmq
브라우저에서 http://localhost:8000 접속
"""

import argparse
import os
import threading
import time
from glob import glob

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

# 설정
MODEL_NAME = "yolo11n-pose.pt"  
ZMQ_PORT = 5555
HTTP_PORT = 8000
PROC_FPS = 5                     # analyzer.py 의 FPS 와 맞출 것
FONT = cv2.FONT_HERSHEY_SIMPLEX


# 추론 루프와 Flask 가 공유하는 최신 상태
state = {"jpg": None, "status": {"detected": False, "metrics": {}, "warnings": []}}
state_lock = threading.Lock()


# 프레임 소스
def folder_frames(path, fps):
    files = sorted(glob(os.path.join(path, "*.jpg")) + glob(os.path.join(path, "*.png")))
    if not files:
        raise SystemExit(f"[folder] {path} 에 이미지가 없습니다.")
    print(f"[folder] {len(files)}장 반복 재생: {path}")
    interval = 1.0 / fps
    while True:
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                yield img
            time.sleep(interval)


def webcam_frames(fps):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("[webcam] 카메라를 열 수 없습니다.")
    print("[webcam] PC 웹캠 사용")
    interval = 1.0 / fps
    try:
        while True:
            ok, img = cap.read()
            if ok:
                yield img
            time.sleep(interval)
    finally:
        cap.release()


def zmq_frames(port):
    import imagezmq
    hub = imagezmq.ImageHub(open_port=f"tcp://*:{port}")
    print(f"[zmq] 포트 {port} 에서 Pi 프레임 수신 대기")
    while True:
        _name, jpg = hub.recv_jpg()
        hub.send_reply(b"OK")
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            yield img


def build_source(spec):
    if spec == "zmq":
        return zmq_frames(ZMQ_PORT)
    if spec == "webcam":
        return webcam_frames(PROC_FPS)
    if spec.startswith("folder:"):
        return folder_frames(spec.split(":", 1)[1], PROC_FPS)
    raise SystemExit(f"알 수 없는 --source: {spec}")


# 영상 위 오버레이 ( ASCII만 )
def draw_overlay(img, status):
    if not status.get("detected"):
        cv2.putText(img, "NO PERSON", (10, 30), FONT, 0.9, (0, 0, 255), 2)
        return
    warnings = status.get("warnings", [])
    banner = "BAD POSTURE" if warnings else "POSTURE OK"
    bcolor = (0, 0, 255) if warnings else (0, 180, 0)
    cv2.putText(img, banner, (10, 30), FONT, 1.0, bcolor, 2)
    y = 62
    for k, m in status.get("metrics", {}).items():
        lo, hi = m["range"]
        flag = {"low": "LOW", "high": "HIGH"}.get(m.get("direction"), "ok")
        txt = f"{k}: {m['value']} [{lo:.0f}~{hi:.0f}] {flag}"
        c = (0, 0, 255) if m["warning"] else (210, 210, 210)
        cv2.putText(img, txt, (10, y), FONT, 0.6, c, 2)
        y += 26


# 추론 루프 ( 백그라운드 스레드 )
def inference_loop(source_gen):
    from ultralytics import YOLO
    from analyzer import Analyzer, keypoints

    print(f"[model] {MODEL_NAME} 로딩...")
    model = YOLO(MODEL_NAME)
    analyzer = Analyzer()
    print("[model] 준비 완료, 추론 시작")

    for frame in source_gen:
        results = model(frame, verbose=False)
        r = results[0]
        kp = keypoints(r)
        if kp is not None:
            out = analyzer.update(kp)
        else:
            out = {"detected": False, "metrics": {}, "warnings": []}

        annotated = r.plot(labels=False, conf=False)          # 스켈레톤이 그려진 BGR 이미지
        draw_overlay(annotated, out)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with state_lock:
                state["jpg"] = buf.tobytes()
                state["status"] = out


# Flask 웹
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# 단순 비디오만 보기
@app.route("/video")
def video():
    def gen():
        while True:
            with state_lock:
                jpg = state["jpg"]
            if jpg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1.0 / PROC_FPS)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# 디버그용 상태 확인
@app.route("/status")
def status():
    with state_lock:
        return jsonify(state["status"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="folder:./debug_frames",
                    help="folder:<경로> | webcam | zmq")
    args = ap.parse_args()

    src = build_source(args.source)
    t = threading.Thread(target=inference_loop, args=(src,), daemon=True)
    t.start()

    print(f"웹 서버: http://localhost:{8020}")
    app.run(host="0.0.0.0", port=8020, threaded=True)


if __name__ == "__main__":
    main()