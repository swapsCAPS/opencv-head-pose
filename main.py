import numpy as np
import os
import cv2 as cv
import dlib
import time
import threading

cap            = cv.VideoCapture(0)
detector       = dlib.get_frontal_face_detector()
predictor_path = "{}/Downloads/shape_predictor_68_face_landmarks.dat".format(os.environ['HOME'])
predictor      = dlib.shape_predictor(predictor_path)

facial_landmark_indices = {
    "eye_left":    46,
    "eye_right":   36,
    "mouth_left":  54,
    "mouth_right": 48,
    "nose_tip":    30,
    "chin":        8,
}

draw_color = dlib.rgb_pixel(0,255,0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

count = 0

landmarks = []

REQUESTED_WIDTH = 320

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    ratio = REQUESTED_WIDTH / float(frame.shape[1])

    new_width  = int(frame.shape[1] * ratio)
    new_height = int(frame.shape[0] * ratio)

    resized_frame = cv.resize(frame, (new_width, new_height), interpolation = cv.INTER_AREA)

    count += 1

    if count >= 5:
        #  resized_frame = cv.resize(frame, )
        detected_faces = detector(resized_frame, 1)

        for face in detected_faces:
            shape     = predictor(frame, face)
            landmarks = map(lambda idx: shape.part(idx) , facial_landmark_indices.values())

        count = 0

    for landmark in landmarks:
        cv.circle(frame, (landmark.x, landmark.y), 3, (0, 255, 0), -1)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
