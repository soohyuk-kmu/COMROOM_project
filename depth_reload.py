import pyrealsense2 as rs      # RealSense SDK
import cv2                     # 화면 출력 및 저장
import numpy as np            # 배열 처리
import os                     # 폴더 생성용
import time                   # 재시도 시간용

# ---------------------------------------------
# 📁 저장 폴더 생성
# ---------------------------------------------
save_dir = "dataset_rgb"
os.makedirs(save_dir, exist_ok=True)
count = 0

# ---------------------------------------------
# 📌 파이프라인 시작(연결 실패 시 무한 재시도)
# ---------------------------------------------
def start_pipeline():
    while True:
        try:
            print("📡 파이프라인 시작 시도중...")

            pipeline = rs.pipeline()
            config = rs.config()

            # 🔥 RGB ONLY 스트림: 1280x720 / 30fps
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

            pipeline.start(config)
            print("✅ RealSense 연결 성공!")
            return pipeline

        except Exception as e:
            print("❌ 연결 실패:", e)
            print("⏳ 5초 후 재시도합니다...")
            time.sleep(5)


# ---------------------------------------------
# 📌 메인 루프 시작
# ---------------------------------------------
pipeline = start_pipeline()

try:
    while True:
        try:
            # 프레임 대기 (5초 타임아웃) 
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()

            if not color_frame:
                raise RuntimeError("⭕ RGB 프레임 없음 (연결 불안정)")

            # numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())

            # 화면 출력
            cv2.imshow("D435i RGB Stream", color_image)

            key = cv2.waitKey(1)

            # ESC 종료
            if key == 27:
                break

            # SPACE → RGB 이미지 저장
            if key == 32:
                file_path = os.path.join(save_dir, f"rgb_{count}.jpg")
                cv2.imwrite(file_path, color_image)
                print(f"💾 Saved: {file_path}")
                count += 1

        except Exception as e:
            # -------------------------------
            # 📌 프레임 오류 → 파이프라인 재시작
            # -------------------------------
            print("\n⚠️ 프레임 오류 발생:", e)
            print("🔄 파이프라인 재시작...\n")

            pipeline.stop()
            pipeline = start_pipeline()

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("🔚 종료 완료!")
