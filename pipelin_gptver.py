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
    "이랑", "랑", "하고", "과", "와",
    "에서", "으로", "로",
    "은", "는", "이", "가", "을", "를", "에"
]

# laptop / notebook 제거한 버전
YOLO_CLASSES = [
    "airpods", "cell phone", "tissue", "mouse",
    "bottle", "glasses", "jelly", "card", "wallet",
    "lipbalm", "remocon", "pen", "applewatch"
]

# 자연어 → YOLO 클래스 매핑
SYNONYMS = {
    "에어팟": "airpods",
    "이어폰": "airpods",

    "핸드폰": "cell phone",
    "휴대폰": "cell phone",
    "폰": "cell phone",

    "티슈": "tissue",
    "휴지": "tissue",

    "마우스": "mouse",

    "물병": "bottle",
    "보틀": "bottle",

    "안경": "glasses",
    "선글라스": "glasses",

    "젤리": "jelly",

    "카드": "card",
    "신용카드": "card",

    "지갑": "wallet",

    "립밤": "lipbalm",
    "립": "lipbalm",

    "리모콘": "remocon",
    "리모컨": "remocon",

    "펜": "pen",
    "볼펜": "pen",

    "애플워치": "applewatch",
    "워치": "applewatch"
}

def split_particle(word: str):
    for p in particles:
        if word.endswith(p):
            return [word[:-len(p)], p]
    return [word]

def remove_particle(word: str):
    for p in particles:
        if word.endswith(p):
            return word[:-len(p)]
    return word

def map_to_class(text: str):
    tokens = []
    for w in text.split():
        tokens.extend(split_particle(w))

    for token in tokens:
        stem = remove_particle(token)
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
# 9분할 그리드 그리기
# =============================================================
def draw_grid(frame):
    h, w = frame.shape[:2]
    color = (0,255,0)
    cv2.line(frame, (w//3, 0), (w//3, h), color, 2)
    cv2.line(frame, (2*w//3, 0), (2*w//3, h), color, 2)
    cv2.line(frame, (0, h//3), (w, h//3), color, 2)
    cv2.line(frame, (0, 2*h//3), (w, 2*h//3), color, 2)
    return frame


# =============================================================
# STT / TTS
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
# YOLO 코드 기본 유지
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
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


# =============================================================
# RealSense 초기화
# =============================================================
def init_realsense() -> Optional[rs.pipeline]:
    if not REALSENSE_AVAILABLE:
        print("❌ pyrealsense2 없음 → RealSense 불가")
        return None
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
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
        if f:
            return np.asanyarray(f.get_data())
    except:
        return None
    return None


# =============================================================
# 메인 실행부
# =============================================================
def main():
    args = parse_args()
    model = YOLO(args.weights)

    use_rs = args.source.lower() in ["realsense","rs","d435i"]
    pipeline = None
    cap = None

    # -----------------------------
    # RealSense: 성공할 때까지 무한 재시도
    # -----------------------------
    if use_rs:
        while pipeline is None:
            print("🔄 RealSense 연결 시도중...")
            pipeline = init_realsense()
            if pipeline is None:
                print("❌ 연결 실패! 5초 후 재시도…")
                time.sleep(5)
        print("✅ RealSense 최종 연결 성공!")

    else:
        cap = cv2.VideoCapture(to_int_if_digit(args.source))

    target_object = None

    # -----------------------------
    # 메인 루프
    # -----------------------------
    try:
        while True:

            if use_rs:
                frame = get_frame_realsense(pipeline)
                ok = frame is not None
            else:
                ok, frame = cap.read()

            if not ok:
                continue

            fh, fw = frame.shape[:2]

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )

            annotated = results[0].plot()
            annotated = draw_grid(annotated)

            key = cv2.waitKey(1) & 0xFF

            # -----------------------------
            # S 키 → 음성 인식
            # -----------------------------
            if key == ord('s'):
                text = stt_listen()
                target_object = map_to_class(text)

                if not target_object:
                    tts_speak("무슨 물건인지 모르겠어요.")
                else:
                    print("🎯 찾는 객체:", target_object)

            # -----------------------------
            # YOLO 탐지에서 물건 찾기
            # -----------------------------
            if target_object:
                boxes = results[0].boxes
                if boxes:
                    for box in boxes:
                        name = results[0].names[int(box.cls[0])]
                        if name == target_object:
                            x1, y1, x2, y2 = box.xyxy[0]
                            cx = (x1+x2)/2
                            cy = (y1+y2)/2

                            region = grid_region(cx, cy, fw, fh)
                            speak_text = f"{target_object}은 {GRID_TEXT.get(region)}"
                            print("📢", speak_text)
                            tts_speak(speak_text)

                            target_object = None
                            break

            if args.show:
                cv2.imshow("YOLO + Grid", annotated)

            if key == 27:
                break

    finally:
        if pipeline:
            pipeline.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()


# =============================================================
if __name__ == "__main__":
    main()
