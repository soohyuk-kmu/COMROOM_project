import argparse
import time
import threading
from queue import Queue
import cv2
import numpy as np
import os
import glob
from collections import deque
from ultralytics import YOLO

# --- Korean fuzzy + jamo
from rapidfuzz import process, fuzz
from jamo import hangul_to_jamo, jamo_to_hangul

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except:
    REALSENSE_AVAILABLE = False

import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment


# ===============================================================
# 🔊 TTS
# ===============================================================
TTS_QUEUE = Queue()

def tts_worker():
    while True:
        text = TTS_QUEUE.get()
        try:
            t = gTTS(text=text, lang="ko")
            t.save("tts.mp3")
            sound = AudioSegment.from_mp3("tts.mp3")
            sound.export("tts.wav", format="wav")
            os.system("ffmpeg -y -i tts.wav -filter:a 'atempo=1.3' tts_fast.wav 2>/dev/null")
            os.system("aplay -q tts_fast.wav")
        except Exception as e:
            print("TTS ERROR:", e)
        TTS_QUEUE.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def tts_speak(t):
    print("🔊:", t)
    TTS_QUEUE.put(t)


# ===============================================================
# 📚 클래스 & 동의어
# ===============================================================
particles = ["이랑","랑","하고","과","와","에서","으로","로","은","는","이","가","을","를","에"]

YOLO_CLASSES = [
   "airpods","cell phone","tissue","mouse",
   "bottle","glasses","jelly","card","wallet",
   "lipbalm","remocon","pen","applewatch"
]

SYNONYMS = {
    "에어팟":"airpods","이어폰":"airpods",
    "핸드폰":"cell phone","휴대폰":"cell phone","폰":"cell phone",
    "티슈":"tissue","휴지":"tissue","화장지":"tissue",
    "마우스":"mouse",
    "물병":"bottle","보틀":"bottle",
    "안경":"glasses","선글라스":"glasses",
    "젤리":"jelly",
    "카드":"card","신용카드":"card",
    "지갑":"wallet","쥐갑":"wallet","지압":"wallet",
    "립밤":"lipbalm","립":"lipbalm",
    "리모콘":"remocon","리모컨":"remocon",
    "펜":"pen","볼펜":"pen",
    "애플워치":"applewatch","애쁠워치":"applewatch",
    "워치":"applewatch"
}


def remove_particle(w):
    for p in particles:
        if w.endswith(p):
            return w[:-len(p)]
    return w

def josa(word):
    last = word[-1]
    jong = (ord(last)-ord("가")) % 28
    return "은" if jong != 0 else "는"


# ===============================================================
# 🔥 발음 보정 (rapidfuzz + jamo)
# ===============================================================
PHONETIC = {
    "ㅂ":"ㅍ","ㅍ":"ㅂ",
    "ㄱ":"ㅋ","ㅋ":"ㄱ",
    "ㅈ":"ㅊ","ㅊ":"ㅈ",
    "ㄹ":"ㄴ","ㄴ":"ㄹ",
    "ㅓ":"ㅗ","ㅗ":"ㅓ",
    "ㅏ":"ㅑ","ㅑ":"ㅏ",
    "ㅜ":"ㅠ","ㅠ":"ㅜ",
    "ㅐ":"ㅔ","ㅔ":"ㅐ",
}

def jamo_correct(text):
    try:
        j = list(hangul_to_jamo(text))
        for i, ch in enumerate(j):
            if ch in PHONETIC:
                j[i] = PHONETIC[ch]
        return jamo_to_hangul(''.join(j))
    except:
        return text

ALL_WORDS = list(SYNONYMS.keys()) + YOLO_CLASSES

def fuzzy_correct(word):
    w1 = jamo_correct(word)
    best, score, _ = process.extractOne(w1, ALL_WORDS, scorer=fuzz.ratio)
    return best if score >= 70 else word


# ===============================================================
# 🔥 복수 매핑
# ===============================================================
def map_to_classes(text):
    found = []
    for w in text.split():
        stem = remove_particle(w)
        corrected = fuzzy_correct(stem)
        if corrected in SYNONYMS:
            found.append((SYNONYMS[corrected], corrected))
        elif corrected in YOLO_CLASSES:
            found.append((corrected, corrected))
    return list(dict.fromkeys(found))


# ===============================================================
# 🗺 28개 구역
# ===============================================================
GRID_NAME = {
    1:"소파 오른쪽 끝", 2:"집 중앙 하단", 3:"집 중앙 하단", 4:"와인셀러 앞",
    5:"소파 앞", 6:"와인셀러와 TV 사이", 7:"소파 앞", 8:"TV 앞",
    9:"소파와 침대 사이", 10:"침대 앞", 11:"서랍장 앞", 12:"TV와 서랍장 사이",

    13:"소파 중앙 - 좌상단", 14:"소파 중앙 - 우상단",
    15:"소파 중앙 - 좌하단", 16:"소파 중앙 - 우하단",

    17:"거실 중앙 - 좌상단", 18:"거실 중앙 - 우상단",
    19:"거실 중앙 - 좌하단", 20:"거실 중앙 - 우하단",

    21:"침대쪽 중앙 - 좌상단", 22:"침대쪽 중앙 - 우상단",
    23:"침대쪽 중앙 - 좌하단", 24:"침대쪽 중앙 - 우하단",

    25:"주방 앞 - 좌상단", 26:"주방 앞 - 우상단",
    27:"주방 앞 - 좌하단", 28:"주방 앞 - 우하단"
}

SUBDIV = {6,7,10,11}
SUBDIV_BASE = {6:13,7:17,10:21,11:25}


def region_16(cx, cy, w, h):
    col = int(cx//(w/4))
    row = int(cy//(h/4))
    return row*4 + col + 1


def region28(cx, cy, w, h):
    r16 = region_16(cx, cy, w, h)

    if r16 not in SUBDIV:
        mapping = {
            1:1, 2:2, 3:3, 4:4, 5:5,
            8:6, 9:7, 12:8,
            13:9, 14:10, 15:11, 16:12
        }
        return mapping.get(r16, 12)

    base = SUBDIV_BASE[r16]

    r = r16 - 1
    row = r // 4
    col = r % 4

    x1 = int(w*col/4); y1 = int(h*row/4)
    x2 = int(w*(col+1)/4); y2 = int(h*(row+1)/4)

    mx = (x1+x2)//2; my = (y1+y2)//2

    horiz = 0 if cx < mx else 1
    vert = 0 if cy < my else 1

    return base + (vert*2 + horiz)


# ===============================================================
# 📸 RealSense
# ===============================================================
def init_rs():
    if not REALSENSE_AVAILABLE:
        print("RealSense 없음!")
        return None
    try:
        p = rs.pipeline()
        c = rs.config()
        c.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        p.start(c)
        print("🐣 RealSense 연결 성공! 반가워요!")
        return p
    except Exception as e:
        print("RealSense Error:", e)
        return None


def get_rs(pipe):
    try:
        f = pipe.wait_for_frames().get_color_frame()
        return np.asanyarray(f.get_data()) if f else None
    except:
        return None


# ===============================================================
# 🎥 이벤트 / 버퍼
# ===============================================================
BUFFER = deque()
missing = {}
last_seen = {}
EVENT_DIR = "events"

def save_event(obj):
    ts = int(time.time())
    folder = f"{EVENT_DIR}/{obj}_{ts}"
    os.makedirs(folder, exist_ok=True)
    for i, (t, fr) in enumerate(BUFFER):
        cv2.imwrite(f"{folder}/{i:03d}.jpg", fr)


def update_event(boxes, w, h):
    present = {b["class"] for b in boxes}
    now = time.time()

    for obj in YOLO_CLASSES:
        if obj in present:
            b = [x for x in boxes if x["class"] == obj][0]
            r = region28(b["cx"], b["cy"], w, h)
            last_seen[obj] = {"loc": GRID_NAME[r], "time": now}
            missing[obj] = 0
        else:
            missing[obj] = missing.get(obj, 0) + 1
            if missing[obj] == 10:
                save_event(obj)


def recent_event(obj):
    lst = sorted(glob.glob(f"{EVENT_DIR}/{obj}_*/"), reverse=True)
    return lst[0] if lst else None


def jpg_to_mp4(folder, fps=10):
    imgs = sorted(glob.glob(folder+"/*.jpg"))
    if not imgs: return None
    h, w = cv2.imread(imgs[0]).shape[:2]
    out = folder.rstrip("/") + ".mp4"
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in imgs:
        vw.write(cv2.imread(f))
    vw.release()
    return out


def last_seen_msg(obj, word):
    if obj not in last_seen:
        return f"{word}{josa(word)} 최근 기록이 없습니다."
    rec = last_seen[obj]
    dt = int(time.time() - rec["time"])
    return f"{word}{josa(word)} {dt}초 전에 {rec['loc']}에서 마지막으로 감지되었어요."
# ===============================================================
# 🖼 PIP overlay
# ===============================================================
def overlay(base, pipf):
    if pipf is None:
        return base
    h, w = base.shape[:2]
    eh, ew = pipf.shape[:2]

    scale = 0.6
    nh = int(h * scale)
    nw = int((ew / eh) * nh)

    pip = cv2.resize(pipf, (nw, nh))

    y1, y2 = 10, 10 + nh
    x1, x2 = 10, 10 + nw
    y2 = min(y2, h)
    x2 = min(x2, w)

    pip = pip[:y2-y1, :x2-x1]
    base[y1:y2, x1:x2] = pip

    return base


# ===============================================================
# 🎤 STT (잡음보정 + fuzzy + 복수 객체)
# ===============================================================
def stt_worker(state):
    r = sr.Recognizer()
    with sr.Microphone() as mic:

        # 주변 소음 보정
        r.adjust_for_ambient_noise(mic, duration=0.2)
        r.dynamic_energy_threshold = True
        r.dynamic_energy_adjustment_ratio = 1.0

        tts_speak("찾을 물건 말해주세요!")

        while True:
            try:
                audio = r.listen(mic, timeout=6, phrase_time_limit=4)
            except:
                tts_speak("다시 말해주세요.")
                continue

            try:
                text = r.recognize_google(audio, language="ko-KR")
                print("🗣 STT:", text)
            except:
                tts_speak("다시 한 번 말씀해주세요.")
                continue

            targets = map_to_classes(text)
            if not targets:
                tts_speak("다시 말씀해주세요.")
                continue

            state["targets"] = targets
            state["running"] = False
            return


# ===============================================================
# MAIN
# ===============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="last.pt")
    p.add_argument("--source", default="rs")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    use_rs = args.source in ["rs", "realsense", "d435i"]

    if use_rs:
        pipe = None
        while pipe is None:
            print("🔄 RealSense 연결 시도중…")
            pipe = init_rs()
            if pipe is None:
                print("❌ 실패… 3초 후 재시도!")
                time.sleep(3)
    else:
        pipe = cv2.VideoCapture(args.source)

    state = {"targets": None, "running": False}
    pip_frames = []
    pip_idx = 0

    os.makedirs(EVENT_DIR, exist_ok=True)

    print("🎉 시스템 준비 완료! 물건을 찾아드릴게요!")
    try:
        while True:

            # ------------------ Frame 입력 ------------------
            if use_rs:
                frame = get_rs(pipe)
                if frame is None:
                    continue
            else:
                ok, frame = pipe.read()
                if not ok:
                    continue

            now = time.time()
            h, w = frame.shape[:2]

            # ------------------ 10초 버퍼 유지 ------------------
            BUFFER.append((now, frame.copy()))
            while BUFFER and now - BUFFER[0][0] > 10:
                BUFFER.popleft()

            # ------------------ YOLO ------------------
            res = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )
            annotated = res[0].plot()

            boxes = []
            for b in res[0].boxes:
                try:
                    cls_id = int(b.cls[0])
                    cname = res[0].names[cls_id]
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    boxes.append({"class": cname, "cx": cx, "cy": cy})
                except:
                    continue

            update_event(boxes, w, h)

            key = cv2.waitKey(1) & 0xFF

            # ------------------ STT 시작 ------------------
            if key == ord('s') and not state["running"]:
                state["running"] = True
                threading.Thread(
                    target=stt_worker,
                    args=(state,),
                    daemon=True
                ).start()

            # ===============================================================
            # 🔥 복수 물건 자연스러운 방식 1 (방식1 완전 머지)
            # ===============================================================
            if state["targets"]:
                targets = state["targets"]

                found_list = []    # [(word, location)]
                missed_list = []   # [“지갑은 ~~초 전에 ~~에서 마지막으로 …”]

                for cls, word in targets:
                    found = False

                    # ---------- 실시간에서 발견 ----------
                    for b in boxes:
                        if b["class"] == cls:
                            r = region28(b["cx"], b["cy"], w, h)
                            loc = GRID_NAME[r]
                            found_list.append((word, loc))
                            found = True
                            break

                    # ---------- 실종 된 경우 ----------
                    if not found:
                        missed_list.append(last_seen_msg(cls, word))

                        # ---- PIP 영상 준비 ----
                        folder = recent_event(cls)
                        pip_frames = []
                        pip_idx = 0

                        if folder:
                            mp4 = folder.rstrip("/") + ".mp4"
                            if not os.path.exists(mp4):
                                mp4 = jpg_to_mp4(folder)

                            cap = cv2.VideoCapture(mp4)
                            while True:
                                ok, f2 = cap.read()
                                if not ok:
                                    break
                                pip_frames.append(f2)
                            cap.release()

                # ========== 발견된 물건들 자연스럽게 말하기 ==========
                if found_list:
                    phrases = []
                    for i, (word, loc) in enumerate(found_list):
                        if i < len(found_list) - 1:
                            phrases.append(f"{word}{josa(word)} {loc}에 있고")
                        else:
                            phrases.append(f"{word}{josa(word)} {loc}에 있어요.")
                    tts_speak(", ".join(phrases))

                # ========== 실종 물건 자연스럽게 말하기 ==========
                if missed_list:
                    if len(missed_list) == 1:
                        tts_speak(missed_list[0])
                    else:
                        tts_speak(" 그리고 ".join(missed_list))

                state["targets"] = None

            # ------------------ PIP 오버레이 ------------------
            if pip_frames:
                annotated = overlay(annotated, pip_frames[pip_idx])
                pip_idx += 1
                if pip_idx >= len(pip_frames):  
                    pip_frames = []
                    pip_idx = 0

            # ------------------ 화면 출력 ------------------
            if args.show:
                cv2.imshow("YOLO", annotated)

            if key == 27:   # ESC
                break

    finally:
        print("🧹 종료 중… 임시 이벤트 파일 정리합니다!")
        # os.system("rm -rf events")  # 원하면 켜기

        if use_rs:
            try:
                pipe.stop()
            except:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
