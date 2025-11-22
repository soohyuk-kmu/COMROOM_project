import argparse
import time
import threading
from queue import Queue

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import os


# ======================================================================
# TTS
# ======================================================================
TTS_QUEUE = Queue()

def tts_worker():
    while True:
        text = TTS_QUEUE.get()
        try:
            t = gTTS(text=text, lang='ko')
            t.save("tts_tmp.mp3")

            sound = AudioSegment.from_mp3("tts_tmp.mp3")    
            sound.export("tts_tmp.wav", format="wav")

            os.system("ffmpeg -y -i tts_tmp.wav -filter:a 'atempo=1.5' tts_tmp_fast.wav 2>/dev/null")
            os.system("aplay -q tts_tmp_fast.wav")

        except Exception as e:
            print("TTS 오류:", e)

        TTS_QUEUE.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def tts_speak(text):
    print("🔊:", text)
    TTS_QUEUE.put(text)


# ======================================================================
# 조사/클래스 매핑
# ======================================================================
particles = [
    "이랑","랑","하고","과","와",
    "에서","으로","로",
    "은","는","이","가","을","를","에"
]

YOLO_CLASSES = [
    "airpods","cell phone","tissue","mouse",
    "bottle","glasses","jelly","card","wallet",
    "lipbalm","remocon","pen","applewatch"
]

SYNONYMS = {
    "에어팟":"airpods",
    "이어폰":"airpods",
    "핸드폰":"cell phone",
    "휴대폰":"cell phone",
    "폰":"cell phone",
    "티슈":"tissue",
    "휴지":"tissue",
    "마우스":"mouse",
    "물병":"bottle",
    "보틀":"bottle",
    "안경":"glasses",
    "선글라스":"glasses",
    "젤리":"jelly",
    "카드":"card",
    "신용카드":"card",
    "지갑":"wallet",
    "립밤":"lipbalm",
    "립":"lipbalm",
    "리모콘":"remocon",
    "리모컨":"remocon",
    "펜":"pen",
    "볼펜":"pen",
    "애플워치":"applewatch",
    "워치":"applewatch"
}

def split_particle(word):
    for p in particles:
        if word.endswith(p):
            return [word[:-len(p)], p]
    return [word]

def remove_particle(word):
    for p in particles:
        if word.endswith(p):
            return word[:-len(p)]
    return word


def josa_eunneun(word: str):
    if not word:
        return "은"
    last = word[-1]
    if "가" <= last <= "힣":
        jong = (ord(last) - ord("가")) % 28
        return "은" if jong != 0 else "는"
    return "는"


def map_to_class(text):
    tokens = []
    for w in text.split():
        tokens.extend(split_particle(w))

    for token in tokens:
        stem = remove_particle(token)

        if stem in SYNONYMS:
            return SYNONYMS[stem], stem
        if stem in YOLO_CLASSES:
            return stem, stem

    return None, None


# ======================================================================
# 16구역 + 세분할
# ======================================================================
SUBDIV_TARGETS = {6, 7, 10, 11}

GRID_TEXT = {
    1:"소파 오른쪽 끝에 있습니다.",
    2:"집 중앙 하단에 있습니다.",
    3:"집 중앙 하단에 있습니다.",
    4:"와인셀러 앞에 있습니다.",
    5:"소파 앞에 있습니다.",
    6:"집 중앙에 있습니다",
    7:"집 중앙에 있습니다",
    8:"와인셀러와 TV 사이에 있습니다.",
    9:"소파 앞에 있습니다.",
    10:"집 중앙에 있습니다.",
    11:"집 중앙에 있습니다.",
    12:"TV 앞에 있습니다.",
    13:"소파와 침대 사이에 있습니다.",
    14:"침대 앞에 있습니다.",
    15:"서랍장 앞에 있습니다.",
    16:"TV와 서랍장 사이에 있습니다."
}

def region_16(cx, cy, w, h):
    col = int(cx // (w / 4))
    row = int(cy // (h / 4))
    col = min(col, 3)
    row = min(row, 3)
    return row * 4 + col + 1

def sub_region_2x2(cx, cy, w, h, region16):
    if region16 not in SUBDIV_TARGETS:
        return None

    r = region16 - 1
    row = r // 4
    col = r % 4

    x1 = int(w * col / 4)
    y1 = int(h * row / 4)
    x2 = int(w * (col + 1) / 4)
    y2 = int(h * (row + 1) / 4)

    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2

    horiz = "왼쪽" if cx < mx else "오른쪽"
    vert  = "위" if cy < my else "아래"

    return f"{horiz} {vert}"


# ======================================================================
# 🔥 STT (안전한 자동 재시도 버전)
# ======================================================================
def stt_thread(state):
    r = sr.Recognizer()
    r.energy_threshold = 300

    tts_speak("어떤 물건을 찾을까요?")

    while True:
        print("🎤 STT 대기중…")

        # --- 녹음 ---
        try:
            with sr.Microphone() as mic:
                audio = r.listen(mic, timeout=5, phrase_time_limit=5)
        except Exception:
            print("❌ 녹음 실패. 다시 시도...")
            tts_speak("다시 말씀해 주세요.")
            time.sleep(0.5)
            continue

        # --- 음성 인식 ---
        try:
            text = r.recognize_google(audio, language='ko-KR')
            print("🗣 인식:", text)
        except Exception:
            print("❌ 음성 인식 실패. 다시 시도...")
            tts_speak("다시 말씀해 주세요.")
            time.sleep(0.5)
            continue

        # --- 클래스 매핑 ---
        cls, user_word = map_to_class(text)

        if not cls:
            print("❌ 매핑 실패. 다시 시도...")
            tts_speak("다시 말씀해 주세요.")
            time.sleep(0.5)
            continue

        # --- 성공 ---
        print(f"🎯 찾는 객체: {cls} (사용자단어: {user_word})")
        state["target"] = cls
        state["user_word"] = user_word
        state["running"] = False
        return  # 스레드 종료


# ======================================================================
# RealSense
# ======================================================================
def init_realsense():
    if not REALSENSE_AVAILABLE:
        print("❌ pyrealsense2 없음")
        return None
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipe.start(cfg)
        print("✅ RealSense 연결 성공!")
        return pipe
    except Exception as e:
        print("❌ RealSense 오류:", e)
        return None

def get_frame_realsense(pipe):
    try:
        frames = pipe.wait_for_frames(timeout_ms=3000)
        f = frames.get_color_frame()
        if f:
            return np.asanyarray(f.get_data())
    except:
        return None
    return None


# ======================================================================
# MAIN
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="last.pt")
    p.add_argument("--source", type=str, default="rs")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    use_rs = args.source.lower() in ["rs", "realsense", "d435i"]
    pipeline = None

    if use_rs:
        while pipeline is None:
            print("🔄 RealSense 연결 시도중")
            pipeline = init_realsense()
            if pipeline is None:
                print("❌ 실패, 5초 후 재시도")
                time.sleep(5)
    else:
        pipeline = cv2.VideoCapture(args.source)

    stt_state = {
        "target": None,
        "user_word": None,
        "running": False
    }

    try:
        while True:

            # 프레임 입력
            if use_rs:
                frame = get_frame_realsense(pipeline)
            else:
                _, frame = pipeline.read()

            if frame is None:
                continue

            h, w = frame.shape[:2]

            results = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False
            )
            annotated = results[0].plot()

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------------------------
            # 🔥 STT 시작 (s 누를 때만)
            # ------------------------------------------------------------------
            if key == ord('s') and not stt_state["running"]:
                stt_state["running"] = True
                threading.Thread(target=stt_thread, args=(stt_state,), daemon=True).start()

            # ------------------------------------------------------------------
            # 🔥 YOLO 타겟 검색
            # ------------------------------------------------------------------
            if stt_state["target"]:
                yolo_target = stt_state["target"]
                user_word = stt_state["user_word"]

                for box in results[0].boxes:
                    cname = results[0].names[int(box.cls[0])]

                    if cname == yolo_target:

                        x1, y1, x2, y2 = box.xyxy[0]
                        cx = float((x1 + x2) / 2)
                        cy = float((y1 + y2) / 2)

                        region16 = region_16(cx, cy, w, h)
                        loc_base = GRID_TEXT.get(region16)

                        # 세분할 판단
                        sub = sub_region_2x2(cx, cy, w, h, region16)

                        if sub:
                            loc_final = f"{region16}번 구역 {sub}에 있습니다."
                        else:
                            loc_final = loc_base

                        josa = josa_eunneun(user_word)
                        tts_speak(f"{user_word}{josa} {loc_final}")

                        stt_state["target"] = None
                        break

            if args.show:
                cv2.imshow("YOLO", annotated)

            if key == 27:
                break

    finally:
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
