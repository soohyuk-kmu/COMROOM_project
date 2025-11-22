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

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except:
    REALSENSE_AVAILABLE = False

import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment


# ===============================================================
# TTS
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
            os.system("ffmpeg -y -i tts.wav -filter:a 'atempo=1.35' tts_fast.wav 2>/dev/null")
            os.system("aplay -q tts_fast.wav")
        except:
            pass
        TTS_QUEUE.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def tts_speak(t):
    print("🔊:", t)
    TTS_QUEUE.put(t)


# ===============================================================
# 단어 매핑
# ===============================================================
particles = ["이랑","랑","하고","과","와","에서","으로","로","은","는","이","가","을","를","에"]

YOLO_CLASSES = [
   "airpods","cell phone","tissue","mouse",
   "bottle","glasses","jelly","card","wallet",
   "lipbalm","remocon","pen","applewatch"
]

SYNONYMS = {
    "에어팟":"airpods", "이어폰":"airpods",
    "핸드폰":"cell phone", "휴대폰":"cell phone", "폰":"cell phone",
    "티슈":"tissue", "휴지":"tissue",
    "마우스":"mouse",
    "물병":"bottle", "보틀":"bottle",
    "안경":"glasses", "선글라스":"glasses",
    "젤리":"jelly",
    "카드":"card", "신용카드":"card",
    "지갑":"wallet",
    "립밤":"lipbalm", "립":"lipbalm",
    "리모콘":"remocon", "리모컨":"remocon",
    "펜":"pen", "볼펜":"pen",
    "애플워치":"applewatch", "워치":"applewatch"
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

def map_to_class(text):
    tokens=text.split()
    for w in tokens:
        stem = remove_particle(w)
        if stem in SYNONYMS:
            return SYNONYMS[stem], stem
        if stem in YOLO_CLASSES:
            return stem, stem
    return None, None


# ===============================================================
# 28개 구역 이름
# ===============================================================
GRID_NAME = {
    1:"소파 오른쪽 끝",
    2:"집 중앙 하단",
    3:"집 중앙 하단",
    4:"와인셀러 앞",
    5:"소파 앞",
    6:"와인셀러와 TV 사이",
    7:"소파 앞",
    8:"TV 앞",
    9:"소파와 침대 사이",
    10:"침대 앞",
    11:"서랍장 앞",
    12:"TV와 서랍장 사이",
    13:"소파 중앙 - 좌상단",
    14:"소파 중앙 - 우상단",
    15:"소파 중앙 - 좌하단",
    16:"소파 중앙 - 우하단",
    17:"거실 중앙 - 좌상단",
    18:"거실 중앙 - 우상단",
    19:"거실 중앙 - 좌하단",
    20:"거실 중앙 - 우하단",
    21:"침대쪽 중앙 - 좌상단",
    22:"침대쪽 중앙 - 우상단",
    23:"침대쪽 중앙 - 좌하단",
    24:"침대쪽 중앙 - 우하단",
    25:"주방 앞 - 좌상단",
    26:"주방 앞 - 우상단",
    27:"주방 앞 - 좌하단",
    28:"주방 앞 - 우하단"
}

SUBDIV = {6,7,10,11}
SUBDIV_BASE = {6:13,7:17,10:21,11:25}


# ===============================================================
# region28 계산
# ===============================================================
def region_16(cx, cy, w, h):
    col = int(cx // (w/4))
    row = int(cy // (h/4))
    col = min(col,3)
    row = min(row,3)
    return row*4 + col + 1

def region28(cx, cy, w, h):
    r16 = region_16(cx, cy, w, h)

    if r16 not in SUBDIV:
        mapping = {
            1:1, 2:2, 3:3, 4:4, 5:5,
            8:6, 9:7,
            12:8, 13:9, 14:10, 15:11, 16:12
        }
        return mapping.get(r16, 12)

    base = SUBDIV_BASE[r16]

    r = r16 - 1
    row = r // 4
    col = r % 4

    x1 = int(w * col / 4)
    y1 = int(h * row / 4)
    x2 = int(w * (col + 1) / 4)
    y2 = int(h * (row + 1) / 4)

    mx = (x1+x2)//2
    my = (y1+y2)//2

    horiz = 0 if cx < mx else 1
    vert = 0 if cy < my else 1
    idx = vert*2 + horiz

    return base + idx


# ===============================================================
# RealSense
# ===============================================================
def init_rs():
    if not REALSENSE_AVAILABLE:
        return None
    try:
        p = rs.pipeline()
        c = rs.config()
        c.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        p.start(c)
        return p
    except:
        return None

def get_rs(pipe):
    try:
        f = pipe.wait_for_frames().get_color_frame()
        return np.asanyarray(f.get_data()) if f else None
    except:
        return None


# ===============================================================
# 이벤트 & 버퍼
# ===============================================================
BUFFER = deque()
missing = {}
last_seen = {}
EVENT_DIR = "events"

def save_event(obj):
    ts = int(time.time())
    folder = f"{EVENT_DIR}/{obj}_{ts}"
    os.makedirs(folder, exist_ok=True)
    for i,(t,fr) in enumerate(BUFFER):
        cv2.imwrite(f"{folder}/{i:03d}.jpg", fr)

def update_event(boxes,w,h):
    present={b["class"] for b in boxes}
    now=time.time()

    for obj in YOLO_CLASSES:
        if obj in present:
            b=[x for x in boxes if x["class"]==obj][0]
            cx,cy=b["cx"],b["cy"]
            r28=region28(cx,cy,w,h)
            last_seen[obj]={"loc":GRID_NAME[r28],"time":now}
            missing[obj]=0
        else:
            missing[obj]=missing.get(obj,0)+1
            if missing[obj]==10:
                save_event(obj)

def recent_event(obj):
    lst = sorted(glob.glob(f"{EVENT_DIR}/{obj}_*/"), reverse=True)
    return lst[0] if lst else None

def jpg_to_mp4(folder,fps=10):
    imgs=sorted(glob.glob(folder+"/*.jpg"))
    if not imgs: return None
    f0=cv2.imread(imgs[0])
    h,w=f0.shape[:2]
    out=folder.rstrip("/")+".mp4"
    vw=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    for f in imgs:
        vw.write(cv2.imread(f))
    vw.release()
    return out

def last_seen_msg(obj,word):
    if obj not in last_seen:
        return f"{word}{josa(word)} 최근 기록이 없습니다."
    rec=last_seen[obj]
    dt=int(time.time()-rec["time"])
    return f"{word}{josa(word)} {dt}초 전에 {rec['loc']}에서 마지막으로 감지되었습니다."


# ===============================================================
# 📌 PIP overlay — 확대 버전
# ===============================================================
def overlay(yolo_f, event_f):
    if event_f is None:
        return yolo_f

    h, w = yolo_f.shape[:2]
    eh, ew = event_f.shape[:2]

    scale = 0.60
    nh = int(h * scale)
    nw = int((ew / eh) * nh)

    pip = cv2.resize(event_f, (nw, nh))

    y1, y2 = 10, 10 + nh
    x1, x2 = 10, 10 + nw

    y2 = min(y2, h)
    x2 = min(x2, w)
    pip = pip[:y2-y1, :x2-x1]

    yolo_f[y1:y2, x1:x2] = pip
    return yolo_f


# ===============================================================
# STT
# ===============================================================
def stt_worker(state):
    r=sr.Recognizer()
    r.energy_threshold=300
    tts_speak("어떤 물건을 찾을까요?")

    while True:
        try:
            with sr.Microphone() as mic:
                audio=r.listen(mic,timeout=5,phrase_time_limit=5)
        except:
            tts_speak("다시 말씀해주세요.")
            continue
        try:
            text=r.recognize_google(audio,language="ko-KR")
        except:
            tts_speak("다시 말씀해주세요.")
            continue

        cls,word = map_to_class(text)
        if not cls:
            tts_speak("다시 말씀해주세요.")
            continue

        state["target"]=cls
        state["word"]=word
        state["running"]=False
        return


# ===============================================================
# MAIN LOOP
# ===============================================================
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--weights",default="last.pt")
    p.add_argument("--source",default="rs")
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--conf",type=float,default=0.5)
    p.add_argument("--iou",type=float,default=0.5)
    p.add_argument("--show",action="store_true")
    return p.parse_args()

def main():
    args=parse_args()
    model=YOLO(args.weights)

    use_rs = args.source in ["rs","d435i","realsense"]
    pipe = init_rs() if use_rs else cv2.VideoCapture(args.source)

    state={"target":None,"word":None,"running":False}
    pip_frames=[]
    pip_idx=0

    os.makedirs(EVENT_DIR,exist_ok=True)

    try:
        while True:

            # ---------------- frame ----------------
            if use_rs:
                frame=get_rs(pipe)
                if frame is None:
                    continue
            else:
                ok,frame=pipe.read()
                if not ok:
                    continue

            now=time.time()
            h,w=frame.shape[:2]

            BUFFER.append((now,frame.copy()))
            while BUFFER and now-BUFFER[0][0]>10:
                BUFFER.popleft()

            # ---------------- YOLO ----------------
            res=model.predict(frame,imgsz=args.imgsz,conf=args.conf,iou=args.iou,verbose=False)
            annotated=res[0].plot()

            boxes=[]
            for b in res[0].boxes:
                cname=res[0].names[int(b.cls[0])]
                x1,y1,x2,y2=b.xyxy[0]
                cx=(x1+x2)/2
                cy=(y1+y2)/2
                boxes.append({"class":cname,"cx":cx,"cy":cy})

            update_event(boxes,w,h)

            key=cv2.waitKey(1)&0xFF

            # ---------------- STT ----------------
            if key==ord('s') and not state["running"]:
                state["running"]=True
                threading.Thread(target=stt_worker,args=(state,),daemon=True).start()

            # ---------------- 찾기 요청 처리 ----------------
            if state["target"]:
                target=state["target"]
                word=state["word"]
                found=False

                for b in res[0].boxes:
                    cname=res[0].names[int(b.cls[0])]
                    if cname==target:
                        found=True
                        x1,y1,x2,y2=b.xyxy[0]
                        cx=(x1+x2)/2
                        cy=(y1+y2)/2
                        r=region28(cx,cy,w,h)
                        loc=GRID_NAME[r]
                        tts_speak(f"{word}{josa(word)} {loc}에 있습니다.")
                        state["target"]=None
                        break

                if not found:
                    tts_speak(last_seen_msg(target,word))

                    folder=recent_event(target)
                    pip_frames=[]
                    pip_idx=0

                    if folder:
                        mp4=folder.rstrip("/")+".mp4"
                        if not os.path.exists(mp4):
                            mp4=jpg_to_mp4(folder)
                        cap=cv2.VideoCapture(mp4)
                        while True:
                            ok,f2=cap.read()
                            if not ok: break
                            pip_frames.append(f2)
                        cap.release()

                state["target"]=None

            # ---------------- PIP 재생 ----------------
            if pip_frames:
                annotated = overlay(annotated, pip_frames[pip_idx])
                pip_idx += 1
                if pip_idx >= len(pip_frames):
                    pip_frames=[]
                    pip_idx=0

            # ---------------- SHOW ----------------
            if args.show:
                cv2.imshow("YOLO",annotated)

            if key==27:
                break

    finally:
        os.system("rm -rf events")
        if use_rs:
            try: pipe.stop()
            except: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
