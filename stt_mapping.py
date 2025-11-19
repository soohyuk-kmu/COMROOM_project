import speech_recognition as sr  # 🎤 음성 → 텍스트 변환 라이브러리

# ---------------------------------------------
# 📌 YOLO 클래스 목록 (에어팟, 휴지, 마우스, 물병, 안경, 립밤,   
# ---------------------------------------------
classes = [
    "airpods", "cell phone", "tissue", "mouse", "laptop", "bottle",
    "glasses", "jelly", "card", "wallet", "lipbalm", "notebook",
    "remocon", "pen", "applewatch"
]

# ---------------------------------------------
# 📌 조사 리스트 (긴 조사 → 짧은 조사 순)
# ---------------------------------------------
particles = [
    "이랑", "랑", "하고", "과", "와",
    "에서", "으로", "로",
    "은", "는", "이", "가", "을", "를", "에"
]

# ---------------------------------------------
# 📌 자연어 → 클래스 매핑 사전
# ---------------------------------------------
mapping_dict = {
    "노트북": "laptop",
    "랩탑": "laptop",

    "노트": "notebook",

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

# ---------------------------------------------
# 📌 조사 단위 분리 함수
#   예: "노트랑" → ["노트", "랑"]
# ---------------------------------------------
def split_particle(word):
    # 입력된 단어를 조사 기준으로 분리할 리스트
    result = [word]

    # 긴 조사 먼저 체크
    for p in particles:
        if word.endswith(p):  # 조사로 끝난다면
            stem = word[: -len(p)]  # 조사 제거한 원형
            return [stem, p]  # ["노트", "랑"] 형태로 반환

    return [word]  # 조사 없으면 그대로 반환

# ---------------------------------------------
# 📌 조사 제거(매핑용)
# ---------------------------------------------
def remove_particle(word):
    for p in particles:
        if word.endswith(p):
            return word[: -len(p)]
    return word

# ---------------------------------------------
# 📌 STT 수행
# ---------------------------------------------
r = sr.Recognizer()

with sr.Microphone() as source:
    print("말하세요...")
    audio = r.listen(source)

try:
    text = r.recognize_google(audio, language="ko-KR")
    print("\n🎤 인식된 문장:", text)

    # -----------------------------------------
    # 📌 1) 문장을 공백 단위로 나눔
    # -----------------------------------------
    raw_words = text.split()

    # -----------------------------------------
    # 📌 2) 한 단어씩 조사 분리
    # -----------------------------------------
    split_words = []  # 조사 포함 단위 토큰 리스트
    for w in raw_words:
        parts = split_particle(w)   # "노트랑" → ["노트", "랑"]
        split_words.extend(parts)   # 리스트 확장하여 저장

    print("\n📌 조사 포함 분리 토큰:")
    for token in split_words:
        print("-", token)

    # -----------------------------------------
    # 📌 3) 매핑되는 클래스 여러 개 찾기
    # -----------------------------------------
    detected_classes = []  # 여러 개 저장

    for token in split_words:  # 조사 포함 토큰들 검사
        clean = remove_particle(token)  # 조사 제거
        if clean in mapping_dict:       # 매핑 가능하다면
            mapped = mapping_dict[clean]
            detected_classes.append(mapped)

    # -----------------------------------------
    # 📌 4) 결과 출력
    # -----------------------------------------
    if detected_classes:
        print("\n✅ 매핑된 클래스(복수 가능):")
        for c in detected_classes:
            print("-", c)
    else:
        print("\n❌ 매핑된 클래스 없음")

except sr.UnknownValueError:
    print("음성 인식 실패")
except sr.RequestError:
    print("STT 서비스 오류")
