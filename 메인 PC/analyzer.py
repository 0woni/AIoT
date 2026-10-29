"""
측면(side-view) 카메라 기준 자세 분석 모듈
카메라는 사람-컴퓨터를 잇는 시선축에 수직으로 설치 -> 사람의 옆모습을 본다
따라서 시상면(sagittal) 자세만 평가한다: 거북목 | 상체 굽음 | 고개 숙임
(좌우 어깨 비대칭은 측면 뷰에서 가려져 측정 불가 -> 제외)

입력: 한 사람의 keypoints, shape (17, 3) = (x, y, confidence)
      COCO 17 포맷 (ultralytics YOLO pose 기본 출력과 동일)

핵심 설계
  - 모든 지표는 각도라 카메라 거리, 사람 키에 거의 무관 (스케일 불변)
  - 한 프레임 키포인트는 노이즈가 크므로 최근 N프레임의 중앙값으로 평활화
    중앙값은 평균과 달리 검출 실패/튐(outlier)에 강하다
  - 임계값은 세팅값일 뿐이다 실제 카메라/설치 각도로 반드시 보정할 것
"""

import math
import statistics
from collections import deque


# 17 keypoint
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


# 설정 (실제 환경에서 보정 대상)
KP_MIN = 0.5          # 키 포인트 신뢰도 임계값 ( 0.5 이상일때만 키 포인트로 지정함 )
FPS = 5                    # 서버가 처리하는 프레임레이트 ( Pi 송신 fps와 맞출 것 )
WINDOW_SEC = 3.0           # 평활화 윈도우 길이 ( 초 )
WINDOW_LEN = max(1, int(WINDOW_SEC * FPS))

# 좋은 자세 허용 범위 (하한, 상한) (도). 이 범위를 벗어나면 나쁜 자세
# 하한 미만 / 상한 초과 는 서로 다른 나쁜 자세를 뜻한다(예: 거북목 vs 목 젖힘)
# 하한 값들은 특히 추정치 -> 실제 환경에서 반드시 보정할 것
GOOD_POSTURE_RANGE = {
    "forward_head": (-8.0, 18.0),  # 기본값 정의 테스트 중 조금씩 수정해서 하이퍼파리미터 최적화 하기 < -10: 목 뒤로 젖힘 / > 18: 거북목
    "torso_slouch": (-17.0,  8.0),  # < -12: 상체 과하게 뒤로 / > 8: 상체 앞으로 굽음
    "head_down":    (  -7.0, 25.0),  # < 8: 턱 치켜듦 / > 32: 고개 숙임  (중립이 ~18°라 한쪽이 양수)
}

# 범위 이탈 시 메시지 (지표, 방향) -> 메시지  방향: low=하한 미만 high=상한 초과
MESSAGES = {
    ("forward_head", "high"): "머리가 앞으로 나와 있어요(거북목) 턱을 살짝 당겨 귀가 어깨 위에 오도록 하세요",
    ("forward_head", "low"):  "고개가 뒤로 젖혀졌어요(목 과신전) 턱을 자연스럽게 내려 정렬을 맞추세요",
    ("torso_slouch", "high"): "상체가 앞으로 굽었어요 골반 위로 등을 곧게 세워 보세요",
    ("torso_slouch", "low"):  "상체가 뒤로 과하게 젖혀졌어요 등받이에서 살짝 떼고 바로 앉아 보세요",
    ("head_down", "high"):    "고개가 너무 숙여졌어요 모니터를 눈높이로 올려 보세요",
    ("head_down", "low"):     "턱이 너무 들렸어요 모니터를 살짝 내리고 시선을 정면으로 두세요",
}


# 기하 툴 (이미지 좌표계: x 오른쪽 +, y 아래 +)
def _facing_sign(nose, ear):
    """사람이 바라보는 방향(+x인지 -x인지) 코가 귀보다 앞쪽에 있는 방향이 정면"""
    return 1.0 if (nose[0] - ear[0]) >= 0 else -1.0


def forward_angle(upper, lower, facing_sign):
    """upper 점이 lower 점보다 앞으로(정면 방향) 얼마나 기울었는지의 각도(도)
    수직으로 곧게 서 있으면 ~0 앞으로 기울수록 +"""
    dx = (upper[0] - lower[0]) * facing_sign   # 정면 방향 성분 (+면 앞)
    dy_up = -(upper[1] - lower[1])             # 위쪽 성분 (+면 upper가 위)
    return math.degrees(math.atan2(dx, dy_up))


def head_pitch(nose, ear, facing_sign):
    """고개가 아래로 숙여진 정도(도) 정면을 보면 작고 숙일수록 +"""
    dx = (nose[0] - ear[0]) * facing_sign      # 정면 방향 (+)
    dy_down = nose[1] - ear[1]                  # 아래 방향 (+면 코가 귀보다 아래)
    return math.degrees(math.atan2(dy_down, abs(dx) + 1e-6))


# 분석기
class Analyzer:
    def __init__(self):
        self.windows = {k: deque(maxlen=WINDOW_LEN) for k in GOOD_POSTURE_RANGE}

    def update(self, kp):
        """프레임 1장의 keypoints (17,3) 를 처리하고 결과 dict 반환

        반환 예:
        {
          "detected": True,
          "metrics": {"forward_head": {"value": 26.6, "range": [-10.0, 18.0], "warning": True, "direction": "high"}, ...},
          "warnings": [{"type": "forward_head", "direction": "high", "message": "...", "value": 26.6}, ...],
        }
        """
        metrics = self._compute(kp)
        result = {"detected": metrics is not None, "metrics": {}, "warnings": []}
        if metrics is None:
            return result

        # 이번 프레임에서 계산된 지표만 윈도우에 누적
        for k, v in metrics.items():
            self.windows[k].append(v)

        # 평활화(중앙값) 후 허용 범위 판정 (범위 이탈 = 나쁜 자세)
        for k, (low, high) in GOOD_POSTURE_RANGE.items():
            w = self.windows[k]
            if not w:
                continue
            smoothed = statistics.median(w)
            if smoothed < low:
                direction = "low"
            elif smoothed > high:
                direction = "high"
            else:
                direction = None
            over = direction is not None
            result["metrics"][k] = {
                "value": round(smoothed, 1),
                "range": [low, high],
                "warning": over,
                "direction": direction,
            }
            if over:
                result["warnings"].append(
                    {"type": k, "direction": direction,
                     "message": MESSAGES[(k, direction)],
                     "value": round(smoothed, 1)}
                )
        return result

    def reset(self):
        for w in self.windows.values():
            w.clear()

    # 내부
    def _compute(self, kp):
        """카메라를 향한 쪽(신뢰도 높은 쪽)을 골라 각도들을 계산
        측정 불가하면 None, 일부만 가능하면 가능한 지표만 담아 반환"""
        # 좌/우 중 신뢰도 합이 큰 쪽 = 카메라를 향한 쪽
        left_conf = kp[L_EAR][2] + kp[L_SHOULDER][2] + kp[L_HIP][2]
        right_conf = kp[R_EAR][2] + kp[R_SHOULDER][2] + kp[R_HIP][2]
        if left_conf >= right_conf:
            ear, sho, hip = kp[L_EAR], kp[L_SHOULDER], kp[L_HIP]
        else:
            ear, sho, hip = kp[R_EAR], kp[R_SHOULDER], kp[R_HIP]

        nose = kp[NOSE]
        fs = _facing_sign(nose, ear) if nose[2] >= KP_MIN else 1.0

        out = {}
        # 거북목: 귀-어깨
        if ear[2] >= KP_MIN and sho[2] >= KP_MIN:
            out["forward_head"] = forward_angle(ear, sho, fs)
        # 상체 굽음: 어깨-골반
        if sho[2] >= KP_MIN and hip[2] >= KP_MIN:
            out["torso_slouch"] = forward_angle(sho, hip, fs)
        # 고개 숙임: 코-귀
        if nose[2] >= KP_MIN and ear[2] >= KP_MIN:
            out["head_down"] = head_pitch(nose, ear, fs)

        return out if out else None


# (선택) ultralytics 결과에서 메인 인물 keypoints 추출 헬퍼
def keypoints(result, min_conf=0.6):
    kpts = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if kpts is None or kpts.data is None or len(kpts.data) == 0:
        return None
    data = kpts.data
    best_i, best_area = -1, 0.0
    for i in range(len(data)):
        if boxes is not None and float(boxes.conf[i]) < min_conf:
            continue  # 신뢰도 낮은 배경 검출 무시하기
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i]]
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_i, best_area = i, area
    return data[best_i].tolist() if best_i >= 0 else None



# 데모: 카메라/YOLO 없이 가짜 키포인트로 동작 확인
if __name__ == "__main__":
    def make_kp(points):
        kp = [[0.0, 0.0, 0.0] for _ in range(17)]
        for idx, (x, y) in points.items():
            kp[idx] = [float(x), float(y), 0.9]
        return kp

    # 정면 +x 방향(오른쪽)을 바라보는 사람 (옆모습)
    good = make_kp({NOSE: (330, 110), L_EAR: (300, 100),
                    L_SHOULDER: (300, 180), L_HIP: (300, 320)})
    fwd_bad = make_kp({NOSE: (410, 140), L_EAR: (380, 100),       # 앞으로 무너짐
                       L_SHOULDER: (340, 180), L_HIP: (290, 320)})
    rev_bad = make_kp({NOSE: (330, 95), L_EAR: (280, 100),        # 반대로 젖혀짐
                       L_SHOULDER: (300, 180), L_HIP: (345, 320)})

    def show(title, sample):
        a = Analyzer()
        for _ in range(WINDOW_LEN):
            res = a.update(sample)
        print(f"\n=== {title} ===")
        tagmap = {"low": "↓하한 이탈", "high": "↑상한 초과", None: "OK"}
        for k, m in res["metrics"].items():
            lo, hi = m["range"]
            print(f"  {k:13s} {m['value']:6.1f}°  허용[{lo:.0f}~{hi:.0f}]  {tagmap[m['direction']]}")
        for w in res["warnings"]:
            print(f"    - [{w['type']}/{w['direction']}] {w['message']}")

    show("바른 자세", good)
    show("앞으로 무너진 자세", fwd_bad)
    show("반대로 젖혀진 자세", rev_bad)