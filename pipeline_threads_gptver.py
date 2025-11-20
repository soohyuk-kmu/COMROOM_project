import argparse
import time
import threading
from typing import Optional

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
# 조사 제거 / 클래스 매핑
# =============================================================
particles = [
    "이랑", "랑", "하고", "과", "와",
    "에서", "으로", "로",
    "은", "는", "이", "가", "을", "를", "에"
]

YOLO_CLASSES = [
    "airpods", "cell phone", "tissue", "mouse",
    "bottle", "glasses", "jelly", "card", "wallet",
    "lipbalm", "remocon", "pen", "applewatch"
]

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


# =============================================================
# 9분할 안내 (텍스트만 사용)
# =============================================================
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
# STT / TTS
# =============================================================
def stt_listen():
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        print("🎤 STT 대기중… 말하세요.")
        audio = r.listen(mic)

    try:
        text = r.recognize_google(audio, language='ko-KR')
        print("🗣 인식:", text)
        return text
    except:
        print("❌ 음성 인식 실패")
        return ""

def tts_speak(text):
    t = gTTS(text=text, lang='ko')
    t.save("tts_out.mp3")
    os.system("mpg123 tts_out.mp3")


# =============================================================
# YOLO
# =============================================================
def parse_args() -> argparse.Namespace:
    from pathlib import Path
    project_root = Path(__file__).parent.absolute()
    default_weights = project_root / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = project_root / "weights" / "yolov8l.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(default_weights))
    parser.add_argument("--source", type=str, default="realsense")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


# =============================================================
# RealSense
# =============================================================
def init_realsense() -> Optional[rs.pipeline]:
    if not REALSENSE_AVAILABLE:
        print("❌ pyrealsense2 없음 → RealSense 불가")
        return None

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        print("✅ RealSense 연결 성공!")
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
# STT 스레드 함수
# =============================================================
def stt_thread(result_holder):
    text = stt_listen()
    cls = map_to_class(text)

    if not cls:
        tts_speak("무슨 물건인지 모르겠어요.")
        result_holder["target"] = None
    else:
        print("🎯 찾는 객체:", cls)
        result_holder["target"] = cls

    result_holder["running"] = False


# =============================================================
# MAIN
# =============================================================
def main():
    args = parse_args()
    model = YOLO(args.weights)

    use_rs = args.source.lower() in ["realsense", "rs", "d435i"]

    pipeline = None
    cap = None

    # RealSense 무한 재시도
    if use_rs:
        while pipeline is None:
            print("🔄 RealSense 연결 시도중…")
            pipeline = init_realsense()
            if pipeline is None:
                print("❌ 실패. 5초 후 재시도…")
                time.sleep(5)
        print("🎉 RealSense 최종 연결 성공!")
    else:
        cap = cv2.VideoCapture(args.source)

    # STT 상태 저장
    stt_state = {"target": None, "running": False}

    try:
        while True:

            # ----- 프레임 -----
            if use_rs:
                frame = get_frame_realsense(pipeline)
                ok = frame is not None
            else:
                ok, frame = cap.read()

            if not ok:
                continue

            fh, fw = frame.shape[:2]

            # ----- YOLO -----
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )
            annotated = results[0].plot()

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------------------
            # S 키 → STT 스레드 시작 (YOLO 멈추지 않음)
            # ------------------------------------------------------------
            if key == ord('s') and not stt_state["running"]:
                print("🎤 STT 스레드 실행")
                stt_state["running"] = True
                threading.Thread(
                    target=stt_thread,
                    args=(stt_state,),
                    daemon=True
                ).start()

            # ------------------------------------------------------------
            # YOLO 탐지 결과로 9분할 안내
            # ------------------------------------------------------------
            target_object = stt_state["target"]

            if target_object:
                boxes = results[0].boxes
                if boxes:
                    for box in boxes:
                        cls_name = results[0].names[int(box.cls[0])]
                        if cls_name == target_object:

                            x1, y1, x2, y2 = box.xyxy[0]
                            cx = float((x1 + x2) / 2)
                            cy = float((y1 + y2) / 2)

                            region = grid_region(cx, cy, fw, fh)
                            speak = f"{target_object}은 {GRID_TEXT.get(region)}"

                            print("📢", speak)
                            tts_speak(speak)

                            stt_state["target"] = None
                            break

            # ----- 화면 출력 -----
            if args.show:
                cv2.imshow("YOLO", annotated)

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
