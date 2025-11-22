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

# RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except:
    REALSENSE_AVAILABLE = False

# TTS + STT
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment


# =========================================================
# 🔊 TTS
# =========================================================
TTS_QUEUE = Queue()

def tts_worker():
    while True:
        text = TTS_QUEUE.get()
        try:
            t = gTTS(text=text, lang="ko")
            t.save("tts_tmp.mp3")
            sound = AudioSegment.from_mp3("tts_tmp.mp3")
            sound.export("tts_tmp.wav", format="wav")
            os.system("ffmpeg -y -i tts_tmp.wav -filter:a 'atempo=1.4' tts_tmp_fast.wav 2>/dev/null")
            os.system("aplay -q tts_tmp_fast.wav")
        except Exception as e:
            print("TTS 오류:", e)
        TTS_QUEUE.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def tts_speak(text):
    print("🔊:", text)
    TTS_QUEUE.put(text)


# =========================================================
# 조사/클래스 매핑
# =========================================================
particles = ["이랑","랑","하고","과","와","에서","으로","로","은","는","이","가","을","를","에"]

YOLO_CLASSES = [
   "airpods","cell phone","tissue","mouse",
   "bottle","glasses","jelly","card","wallet",
   "lipbalm","remocon","pen","applewatch"
]

SYNONYMS = {
    "에어팟":"airpods","이어폰":"airpods",
    "핸드폰":"cell phone","휴대폰":"cell phone","폰":"cell phone",
    "티슈":"tissue","휴지":"tissue",
    "마우스":"mouse",
    "물병":"bottle","보틀":"bottle",
    "안경":"glasses","선글라스":"glasses",
    "젤리":"jelly",
    "카드":"card","신용카드":"card",
    "지갑":"wallet",
    "립밤":"lipbalm","립":"lipbalm",
    "리모콘":"remocon","리모컨":"remocon",
    "펜":"pen","볼펜":"pen",
    "애플워치":"applewatch","워치":"applewatch"
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

def josa_eunneun(word):
    last = word[-1]
    jong = (ord(last) - ord("가")) % 28
    return "은" if jong != 0 else "는"

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


# =========================================================
# 16구역 매핑 + 세분할
# =========================================================
GRID_TEXT = {
    1:"소파 오른쪽 끝에 있습니다.",
    2:"집 중앙 하단에 있습니다.",
    3:"집 중앙 하단에 있습니다.",
    4:"와인셀러 앞에 있습니다.",
    5:"소파 앞에 있습니다.",
    6:"집 중앙에 있습니다.",
    7:"집 중앙에 있습니다.",
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

SUBDIV_TARGETS = {6,7,10,11}

def region_16(cx, cy, w, h):
    col = int(cx // (w/4))
    row = int(cy // (h/4))
    return row*4 + col + 1

def sub_region_2x2(cx, cy, w, h, r16):
    if r16 not in SUBDIV_TARGETS:
        return None
    r = r16 - 1
    row = r // 4
    col = r % 4
    x1 = int(w*col/4); y1 = int(h*row/4)
    x2 = int(w*(col+1)/4); y2 = int(h*(row+1)/4)
    mx = (x1+x2)//2; my = (y1+y2)//2
    horiz = "왼쪽" if cx < mx else "오른쪽"
    vert  = "위" if cy < my else "아래"
    return f"{horiz} {vert}"


# =========================================================
# RealSense
# =========================================================
def init_realsense():
    if not REALSENSE_AVAILABLE:
        return None
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30)
        pipe.start(cfg)
        print("🎥 RealSense 연결 성공")
        return pipe
    except:
        return None

def get_frame_realsense(pipe):
    try:
        frames = pipe.wait_for_frames(timeout_ms=2000)
        f = frames.get_color_frame()
        return np.asanyarray(f.get_data()) if f else None
    except:
        return None


# =========================================================
# 이벤트 시스템 (버퍼 + 실종 감지)
# =========================================================
FRAME_BUFFER = deque()
last_seen = {}
missing_counter = {}

def save_event_clip(obj):
    ts = int(time.time())
    save_dir = f"events/{obj}_{ts}"
    os.makedirs(save_dir, exist_ok=True)

    for idx, (t, f) in enumerate(FRAME_BUFFER):
        cv2.imwrite(f"{save_dir}/{idx:03d}.jpg", f)

    print(f"📁 실종 이벤트 저장: {save_dir}")

def update_event(yolo_boxes, w, h):
    present = {b["class"] for b in yolo_boxes}

    for obj in YOLO_CLASSES:

        if obj in present:
            box = [b for b in yolo_boxes if b["class"] == obj][0]
            cx, cy = box["cx"], box["cy"]
            r16 = region_16(cx, cy, w, h)
            sub = sub_region_2x2(cx, cy, w, h, r16)
            loc = f"{r16}번 구역 {sub}" if sub else GRID_TEXT.get(r16)
            last_seen[obj] = {"loc": loc, "time": time.time()}
            missing_counter[obj] = 0
            continue

        missing_counter[obj] = missing_counter.get(obj, 0) + 1

        if missing_counter[obj] == 10:
            print(f"⚠ 실종 이벤트 발생: {obj}")
            save_event_clip(obj)


def find_latest_event_folder(obj):
    lst = sorted(glob.glob(f"events/{obj}_*/"), reverse=True)
    return lst[0] if lst else None


# =========================================================
# JPG → MP4
# =========================================================
def jpgs_to_mp4(event_dir, fps=10):
    imgs = sorted(glob.glob(event_dir+"/*.jpg"))
    if not imgs:
        return None
    f0 = cv2.imread(imgs[0])
    h, w = f0.shape[:2]
    out_path = event_dir.rstrip("/") + ".mp4"
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in imgs:
        vw.write(cv2.imread(f))
    vw.release()
    return out_path


# =========================================================
# 마지막 위치 안내 메시지
# =========================================================
def get_last_seen_message(obj, word):
    if obj not in last_seen:
        return f"{word}{josa_eunneun(word)} 최근 기록이 없습니다."
    rec = last_seen[obj]
    dt = int(time.time() - rec["time"])
    loc = rec["loc"].replace("에 있습니다.","").replace("있습니다.","")
    return f"{word}{josa_eunneun(word)} {dt}초 전에 {loc}에서 마지막으로 감지되었습니다."


# =========================================================
# PIP 오버레이 (1:1 비율)
# =========================================================
def overlay_event(yolo_frame, event_frame):
    if event_frame is None:
        return yolo_frame
    h, w = yolo_frame.shape[:2]
    eh, ew = event_frame.shape[:2]

    y1, y2 = 10, 10 + eh
    x1, x2 = 10, 10 + ew

    if y2 > h:
        y2 = h
    if x2 > w:
        x2 = w

    pip = event_frame[:y2-y1, :x2-x1]
    yolo_frame[y1:y2, x1:x2] = pip
    return yolo_frame


# =========================================================
# STT
# =========================================================
def stt_thread(state):
    r = sr.Recognizer()
    r.energy_threshold = 300
    tts_speak("어떤 물건을 찾을까요?")

    while True:
        try:
            with sr.Microphone() as mic:
                audio = r.listen(mic, timeout=5, phrase_time_limit=5)
        except:
            tts_speak("다시 말씀해주세요.")
            continue

        try:
            text = r.recognize_google(audio, language="ko-KR")
        except:
            tts_speak("다시 말씀해주세요.")
            continue

        cls, word = map_to_class(text)
        if not cls:
            tts_speak("다시 말씀해주세요.")
            continue

        state["target"] = cls
        state["word"] = word
        state["running"] = False
        return


# =========================================================
# MAIN LOOP
# =========================================================
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

    use_rs = args.source in ["rs","realsense","d435i"]
    pipeline = init_realsense() if use_rs else cv2.VideoCapture(args.source)

    stt_state = {"target":None,"word":None,"running":False}

    pip_frames = []
    pip_idx = 0

    try:
        while True:

            # ---------------- 프레임 입력 ----------------
            frame = None
            if use_rs:
                for _ in range(5):
                    frame = get_frame_realsense(pipeline)
                    if frame is not None:
                        break
                if frame is None:
                    continue
            else:
                ok, frame = pipeline.read()
                if not ok:
                    continue

            h, w = frame.shape[:2]

            # ---------------- 10초 버퍼 유지 ----------------
            now = time.time()
            FRAME_BUFFER.append((now, frame.copy()))
            while FRAME_BUFFER and now - FRAME_BUFFER[0][0] > 10:
                FRAME_BUFFER.popleft()

            # ---------------- YOLO ----------------
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
            annotated = results[0].plot()

            # ---------------- 이벤트 업데이트 ----------------
            yolo_boxes = []
            for box in results[0].boxes:
                cname = results[0].names[int(box.cls[0])]
                x1,y1,x2,y2 = box.xyxy[0]
                cx=(x1+x2)/2; cy=(y1+y2)/2
                yolo_boxes.append({"class":cname,"cx":cx,"cy":cy})

            update_event(yolo_boxes, w, h)

            # ---------------- STT 호출 ----------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not stt_state["running"]:
                stt_state["running"] = True
                threading.Thread(target=stt_thread, args=(stt_state,), daemon=True).start()

            # ---------------- “OO 어디 있어?” 처리 ----------------
            if stt_state["target"]:
                target = stt_state["target"]
                word = stt_state["word"]
                found = False

                # 1) YOLO에서 직접 찾기
                for box in results[0].boxes:
                    cname = results[0].names[int(box.cls[0])]
                    if cname == target:
                        found = True
                        x1,y1,x2,y2 = box.xyxy[0]
                        cx=(x1+x2)/2; cy=(y1+y2)/2
                        r16 = region_16(cx,cy,w,h)
                        sub = sub_region_2x2(cx,cy,w,h,r16)
                        loc = f"{r16}번 구역 {sub}" if sub else GRID_TEXT.get(r16)
                        tts_speak(f"{word}{josa_eunneun(word)} {loc}")
                        stt_state["target"]=None
                        break

                # 2) 못 찾으면 → 마지막 이벤트 영상 로드(PIP, 1회 재생)
                if not found:
                    tts_speak(get_last_seen_message(target, word))

                    folder = find_latest_event_folder(target)
                    pip_frames = []
                    pip_idx = 0

                    if folder:
                        mp4 = folder.rstrip("/")+ ".mp4"
                        if not os.path.exists(mp4):
                            mp4 = jpgs_to_mp4(folder)

                        cap2 = cv2.VideoCapture(mp4)
                        while True:
                            ret, f2 = cap2.read()
                            if not ret:
                                break
                            pip_frames.append(f2)
                        cap2.release()

                stt_state["target"] = None

            # ---------------- PIP 오버레이 (딱 1번만 재생) ----------------
            if pip_frames:
                annotated = overlay_event(annotated, pip_frames[pip_idx])
                pip_idx += 1

                # 🔥 1회 재생 후 자동 종료
                if pip_idx >= len(pip_frames):
                    pip_frames = []
                    pip_idx = 0

            # ---------------- 화면 출력 ----------------
            if args.show:
                cv2.imshow("YOLO", annotated)

            if key == 27:
                break

    finally:
        # 종료 시 이벤트 폴더 전체 삭제
        if os.path.exists("events"):
            os.system("rm -rf events")
            print("🧹 events 폴더 전체 삭제 완료")

        if use_rs:
            try: pipeline.stop()
            except: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
