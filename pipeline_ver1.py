import argparse
import time
from typing import Union, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# RealSense import
# -----------------------------
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("경고: pyrealsense2가 설치되지 않았습니다. RealSense 카메라를 사용할 수 없습니다.")

# -----------------------------
# STT / TTS
# -----------------------------
import speech_recognition as sr
from gtts import gTTS
import os


# =============================================================
# 0. 조사 제거 / 클래스 매핑 / 9분할 위치 텍스트
# =============================================================
particles = [
    "이랑","랑","하고","과","와",
    "에서","으로","로",
    "은","는","이","가","을","를","에"
]

def strip_particle(word):
    for p in particles:
        if word.endswith(p):
            return word[:-len(p)]
    return word

YOLO_CLASSES = [
    "airpods","cell phone","tissue","mouse","laptop","bottle",
    "glasses","jelly","card","wallet","lipbalm","notebook",
    "remocon","pen","carkey"
]

SYNONYMS = {
    "핸드폰": "cell phone",
    "폰": "cell phone",
    "휴지": "tissue",
    "노트북": "laptop",
    "책": "notebook",
    "공책": "notebook",
    "리모콘": "remocon"
}

def map_to_class(text: str):
    words = text.split()
    for w in words:
        stem = strip_particle(w)
        if stem in SYNONYMS:
            return SYNONYMS[stem]
        if stem in YOLO_CLASSES:
            return stem
    return None


# -----------------------------
# 9분할 위치 문구
# -----------------------------
GRID_TEXT = {
    1: "TV와 서랍장 앞에 있습니다.",
    2: "서랍장과 침대 사이에 있습니다.",
    3: "침대와 소파 앞에 있습니다.",
    4: "TV 앞에 있습니다.",
    5: "정가운데에 있습니다.",
    6: "소파 앞에 있습니다.",
    7: "와인셀러 앞에 있습니다.",
    8: "중앙 아래쪽에 있습니다.",
    9: "소파 왼쪽 앞에 있습니다."
}

def grid_region(cx, cy, w, h):
    col = int(cx // (w/3))
    row = int(cy // (h/3))
    return row * 3 + col + 1


# =============================================================
# 1. 시각화용 9분할 그리드
# =============================================================
def draw_grid(frame):
    h, w = frame.shape[:2]
    w1 = w // 3
    w2 = 2 * w // 3
    h1 = h // 3
    h2 = 2 * h // 3

    color = (0,255,0)
    cv2.line(frame, (w1, 0), (w1, h), color, 2)
    cv2.line(frame, (w2, 0), (w2, h), color, 2)
    cv2.line(frame, (0, h1), (w, h1), color, 2)
    cv2.line(frame, (0, h2), (w, h2), color, 2)
    return frame


# =============================================================
# 2. STT / TTS
# =============================================================
def stt_listen():
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        print("🎤 STT 대기중... 말하세요.")
        audio = r.listen(mic)

    try:
        text = r.recognize_google(audio, language='ko-KR')
        print("🗣️ 인식:", text)
        return text
    except:
        print("❌ 음성 인식 실패")
        return ""

def tts_speak(text):
    t = gTTS(text=text, lang='ko')
    t.save("tts_out.mp3")
    os.system("mpg123 tts_out.mp3")


# =============================================================
# 3. 너가 준 YOLO 코드 기반 그대로 유지
# =============================================================
def parse_args() -> argparse.Namespace:
    from pathlib import Path
    
    project_root = Path(__file__).parent.absolute()
    default_weights = project_root / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = project_root / "weights" / "yolov8l.pt"
    
    parser = argparse.ArgumentParser(description="YOLOv8 실시간 추론")

    parser.add_argument("--weights", type=str, default=str(default_weights))
    parser.add_argument("--source", type=str, default="realsense")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fps", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--homography", type=str, default="")
    return parser.parse_args()


def to_int_if_digit(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def load_homography(path: str) -> Union[np.ndarray, None]:
    if not path:
        return None
    try:
        if path.lower().endswith(".npy"):
            H = np.load(path)
        else:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            H = np.asarray(data.get("H"), dtype=np.float64)
        if H.shape == (3,3):
            return H
    except:
        pass
    print("호모그래피 파일을 불러오지 못했습니다.")
    return None


# -----------------------------
# RealSense 초기화
# -----------------------------
def init_realsense() -> Optional[rs.pipeline]:
    if not REALSENSE_AVAILABLE:
        print("❌ pyrealsense2가 없음")
        return None
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8,30)
        config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16,30)
        pipeline.start(config)
        print("✅ RealSense 연결 성공")
        return pipeline
    except Exception as e:
        print("❌ RealSense 연결 실패:", e)
        return None


def get_frame_realsense(pipeline):
    try:
        frames = pipeline.wait_for_frames()
        f = frames.get_color_frame()
        if f: return np.asanyarray(f.get_data())
    except:
        pass
    return None


# =============================================================
# 4. 메인
# =============================================================
def main():
    args = parse_args()

    model = YOLO(args.weights)
    H = load_homography(args.homography)

    # -----------------------------
    # 카메라 선택
    # -----------------------------
    use_rs = args.source.lower() in ["realsense","rs","d435i"]
    pipeline = None
    cap = None

    if use_rs:
        pipeline = init_realsense()
        if pipeline is None:
            print("⚠️ RealSense 사용 불가 → 웹캠으로 전환")
            use_rs = False
            cap = cv2.VideoCapture(0)
    else:
        source = to_int_if_digit(args.source)
        cap = cv2.VideoCapture(source)

    # -----------------------------
    # VideoWriter 준비
    # -----------------------------
    writer = None
    prev_time = time.time()
    initialized_size = False
    frame_idx = 0

    target_object = None  # STT로 요청된 YOLO 클래스

    # =============================================================
    # 루프 시작
    # =============================================================
    try:
        while True:

            # -----------------------------
            # 프레임 얻기
            # -----------------------------
            if use_rs:
                frame = get_frame_realsense(pipeline)
                ok = frame is not None
            else:
                ok, frame = cap.read()

            if not ok or frame is None:
                break

            frame_idx += 1
            fh, fw = frame.shape[:2]

            # 9분할 그리드 시각화
            frame_show = draw_grid(frame.copy())

            # -----------------------------
            # YOLO Predict
            # -----------------------------
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False
            )
            annotated = results[0].plot()

            # -----------------------------
            # S 키 → STT 실행
            # -----------------------------
            key = cv2.waitKey(1)
            if key == ord('s'):
                text = stt_listen()
                target_object = map_to_class(text)

                if not target_object:
                    tts_speak("무슨 물건인지 모르겠어요.")
                else:
                    print("🎯 찾는 객체:", target_object)

            # -----------------------------
            # YOLO 내부에서 target_object 찾기
            # -----------------------------
            if target_object:
                det_boxes = results[0].boxes
                if det_boxes is not None:
                    for box in det_boxes:
                        name = results[0].names[int(box.cls[0])]
                        if name == target_object:
                            x1, y1, x2, y2 = box.xyxy[0]
                            cx = (x1+x2)/2
                            cy = (y1+y2)/2

                            region = grid_region(cx, cy, fw, fh)
                            speak_text = f"{target_object}은 {GRID_TEXT[region]}"
                            print("📢", speak_text)
                            tts_speak(speak_text)

                            target_object = None
                            break

            # -----------------------------
            # 화면 출력
            # -----------------------------
            if args.show:
                cv2.imshow("YOLO + Grid", annotated)

            if key == 27:
                break

    finally:
        if pipeline is not None:
            pipeline.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


# =============================================================
if __name__ == "__main__":
    main()
