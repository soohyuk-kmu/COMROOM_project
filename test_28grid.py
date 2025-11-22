import cv2
import numpy as np

# 4x4 큰 구역
BIG_ROWS = 4
BIG_COLS = 4

# 세분할 되는 구역들
SUBDIV = {6, 7, 10, 11}

def draw_real_28_grid(frame):
    h, w = frame.shape[:2]
    cell_w = w // BIG_COLS
    cell_h = h // BIG_ROWS

    # 큰 4x4 라인 그리기
    for c in range(1, BIG_COLS):
        x = c * cell_w
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 2)

    for r in range(1, BIG_ROWS):
        y = r * cell_h
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 2)

    # 번호 붙이기 (28칸)
    num = 1
    for r in range(BIG_ROWS):
        for c in range(BIG_COLS):
            big_idx = r * 4 + c + 1
            x1 = c * cell_w
            y1 = r * cell_h

            # 일반 구역일 때
            if big_idx not in SUBDIV:
                cv2.putText(frame, str(num), (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                num += 1

            else:
                # 세분할 구역 (2×2)
                half_w = cell_w // 2
                half_h = cell_h // 2

                for sr in range(2): 
                    for sc in range(2):
                        sx = x1 + sc * half_w
                        sy = y1 + sr * half_h
                        ex = sx + half_w
                        ey = sy + half_h

                        cv2.rectangle(frame, (sx, sy), (ex, ey), (200, 200, 200), 1)
                        cv2.putText(frame, str(num), (sx + 10, sy + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 200, 255), 2)
                        num += 1

    return frame


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = draw_real_28_grid(frame)
        cv2.imshow("28 Grid View (REAL)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
