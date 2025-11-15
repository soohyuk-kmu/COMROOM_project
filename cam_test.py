import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
pipeline.start()

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        color = np.asanyarray(frames.get_color_frame().get_data())
        cv2.imshow('Color', color)
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
