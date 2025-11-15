from gtts import gTTS
from playsound import playsound

# 변환할 텍스트
text = "sexy"

# 한국어 설정으로 음성 생성
tts = gTTS(text=text, lang='ko')

# 음성 파일로 저장
tts.save("output.mp3")

# 재생
print("🔊 음성 재생 중...")
playsound("output.mp3")
