# 🧠 COMROOM Project  
💡 YOLO 기반 책상 물건 자동 탐지 & 음성 안내 시스템

---

##  프로젝트 소개
COMROOM Project는 책상 위에 있는 물건을 실시간으로 탐지하고  
사용자가 음성으로 “노트북 어디 있어?”라고 물어보면  
카메라 화면에서 해당 물건을 찾아 강조 및 위치 안내 

## 🚀 주요 기능

### 1.  YOLO기반 객체 탐지
- Roboflow에서 직접 라벨링한 15개 클래스 사용  
airpods, cell phone, tissue, mouse, laptop,
bottle, glasses, jelly, card, wallet,
lipbalm, notebook, remocon, pen, applewatch


### 2. 음성 인식(STT)
- 마이크 입력을 받아  
사용자의 발화를 텍스트로 변환  
- 물건명과 자연어 매핑  
- 예: “랩탑 어디 있어?” → “laptop”

### 3.  음성 출력(TTS)
- 탐지된 물체의 좌표 기반(모형 집 제작 예정)  
- “노트북은 화면 화장실에 있습니다”  
와 같이 TTS로 안내

### 4.  시각화
- 탐지된 물체에 box 표시  


---

