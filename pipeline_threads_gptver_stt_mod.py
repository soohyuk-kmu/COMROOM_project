import argparse
import time
import threading
from typing import Optional
from queue import Queue

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
from pydub import AudioSegment
import os


# =============================================================
# TTS 큐 시스템 (wav + aplay + 빠른 속도 atempo)
# =============================================================
TTS_QUEUE = Queue()

def tts_worker():
    """빠른 TTS (속도 1.5배)"""
    while True:
        text = TTS_QUEUE.get()
        try:
            # gTTS → mp3
            t = gTTS(text=text, lang='ko')
            t.save("tts_tmp.mp3")

            # mp3 → wav
            sound = AudioSegment.from_mp3("tts_tmp.mp3")
            sound.export("tts_tmp.wav", format="wav")

            # wav 속도 빠르게 (1.5배)
            os.system("ffmpeg -y -i tts_tmp.wav -filter:a 'atempo=1.5' tts_tmp_fast.wav 2>/dev/null")

            # 재생
            os.system("aplay -q tts_tmp_fast.wav")

        except Exception as e:
            print("TTS 오류:", e)

        TTS_QUEUE.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def tts_speak(text):
    print("🔊 TTS 요청:", text)
    TTS_QUEUE.put(text)


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
# 🔥 16분할 안내 (4×4)
# =============================================================
GRID_TEXT = {
    1: "1구역에 있습니다.",
    2: "2구역에 있습니다.",
    3: "3구역에 있습니다.",
    4: "4구역에 있습니다.",
    5: "5구역에 있습니다.",
    6: "6구역에 있습니다.",
    7: "7구역에 있습니다.",
    8: "8구역에 있습니다.",
    9: "9구역에 있습니다.",
    10: "10구역에 있습니다.",
    11: "11구역에 있습니다.",
    12: "12구역에 있습니다.",
    13: "13구역에 있습니다.",
    14: "14구역에 있습니다.",
    15: "15구역에 있습니다.",
    16: "16구역에 있습니다."
}

def grid_region(cx, cy, w, h):
    col = int(cx // (w / 4))   # 0~3
    row = int(cy // (h / 4))   # 0~3
    col = min(col, 3)
    row = min(row, 3)
    return row * 4 + col + 1   # 1~16


# =============================================================
# STT
# =============================================================
def stt_listen():
    r = sr.Recognizer()
    r.energy_threshold = 300

    with sr.Microphone() as mic:
        print("🎤 STT 대기중… 말하세요.")
        try:
            audio = r.listen(mic, timeout=5, phrase_time_limit=5)
        except:
            return ""
    try:
        text = r.recognize_google(audio, language='ko-KR')
        print("🗣 인식:", text)
        return text
    except:
        print("❌ 음성 인식 실패")
        return ""


# =============================================================
# STT 스레드 (자동 재시작 버전)
# =============================================================
def stt_thread(state):
    text = stt_listen()
    cls = map_to_class(text)

    if not text.strip():
        tts_speak("다시 말씀해 주세요.")
        state["target"] = None
        state["running"] = True
        state["retry"] = True
        threading.Thread(target=stt_thread, args=(state,), daemon=True).start()
        return

    if not cls:
        tts_speak("무슨 물건인지 모르겠어요. 다시 말씀해 주세요.")
        state["target"] = None
        state["running"] = True
        state["retry"] = True
        threading.Thread(target=stt_thread, args=(state,), daemon=True).start()
        return

    print("🎯 찾는 객체:", cls)
    state["target"] = cls
    state["running"] = False
    state["retry"] = False
    state["searched_before"] = True



# =============================================================
# YOLO Args
# =============================================================
def parse_args():
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
def init_realsense():
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
        frames = pipeline.wait_for_frames(timeout_ms=3000)
        f = frames.get_color_frame()
        if f:
            return np.asanyarray(f.get_data())
    except:
        return None
    return None



# =============================================================
# MAIN LOOP
# =============================================================
def main():
    args = parse_args()
    model = YOLO(args.weights)

    use_rs = args.source.lower() in ["realsense", "rs", "d435i"]
    pipeline = None
    cap = None

    # RealSense 연결 반복
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

    stt_state = {
        "target": None,
        "running": False,
        "retry": False,
        "searched_before": False
    }

    try:
        while True:

            # 프레임 가져오기
            if use_rs:
                frame = get_frame_realsense(pipeline)
                ok = frame is not None
            else:
                ok, frame = cap.read()
            if not ok:
                continue

            fh, fw = frame.shape[:2]

            # YOLO 추론
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )
            annotated = results[0].plot()

            key = cv2.waitKey(1) & 0xFF

            # =====================================================
            # S → STT 최초 1회 실행
            # =====================================================
            if key == ord('s'):
                if not stt_state["running"]:
                    tts_speak("어떤 물건을 찾을까요?")
                    stt_state["running"] = True
                    threading.Thread(target=stt_thread, args=(stt_state,), daemon=True).start()

            # =====================================================
            # YOLO 결과 → 물건 위치 안내 (16분할)
            # =====================================================
            target_object = stt_state["target"]

            if target_object:
                for box in results[0].boxes:
                    cls_name = results[0].names[int(box.cls[0])]
                    if cls_name == target_object:

                        x1, y1, x2, y2 = box.xyxy[0]
                        cx = float((x1 + x2) / 2)
                        cy = float((y1 + y2) / 2)

                        region = grid_region(cx, cy, fw, fh)
                        location_text = GRID_TEXT.get(region)

                        tts_speak(f"{target_object}은 {location_text}")

                        stt_state["target"] = None
                        stt_state["running"] = False
                        break

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
