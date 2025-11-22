#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
import time

def main():
    print("📡 RealSense 초기화 중…")

    pipeline = rs.pipeline()
    config = rs.config()

    # ===========================
    # ✔ RGB 스트림만 활성화
    # ===========================
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print("❌ RealSense 시작 실패:", e)
        return

    print("✅ RealSense 연결 성공! (RGB only mode)")

    try:
        while True:

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # numpy 변환
            color = np.asanyarray(color_frame.get_data())

            # 화면 출력 (RGB only)
            cv2.imshow("RGB 1920x1080", color)

            key = cv2.waitKey(1) & 0xFF

            # 📸 스페이스바 → RGB만 저장
            if key == 32:  # space
                ts = int(time.time())
                rgb_name = f"rgb_{ts}.jpg"
                cv2.imwrite(rgb_name, color)
                print(f"📸 RGB 저장됨 → {rgb_name}")

            # ESC 종료
            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("🛑 종료 완료")


if __name__ == "__main__":
    main()

