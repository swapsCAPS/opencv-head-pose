import numpy as np
import os
import cv2 as cv
import dlib
import time

cap      = cv.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor_path = "{}/Downloads/shape_predictor_68_face_landmarks.dat".format(os.environ['HOME'])
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

draw_color = dlib.rgb_pixel(255,0,0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    detected_faces = detector(frame, 1)

    objects = list(map(lambda pos: predictor(frame, pos), detected_faces))

    win.clear_overlay()
    win.set_image(frame)
    win.add_overlay(objects=objects)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
