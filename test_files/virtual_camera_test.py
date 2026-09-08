import cv2
import numpy as np
import pyvirtualcam

camera = cv2.VideoCapture(0)

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

with pyvirtualcam.Camera(
    width=width,
    height=height,
    fps=fps
) as cam:

    print("Virtual Camera Started")

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cam.send(frame)
        cam.sleep_until_next_frame()