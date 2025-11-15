import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("🎙️ 말하세요...")
    audio = r.listen(source, timeout=5, phrase_time_limit=5)  # 최대 5초까지 대기
    print("녹음 완료! Google API로 전송 중...")

try:
    text = r.recognize_google(audio, language="ko-KR", show_all=False)
    print("✅ 인식된 내용:", text)
except sr.RequestError as e:
    print(f"🌐 Google API 연결 실패: {e}")
except sr.UnknownValueError:
    print("❌ 음성 인식 실패 (음성은 감지됐지만 텍스트 변환 불가)")
except Exception as e:
    print(f"⚠️ 기타 오류: {e}")