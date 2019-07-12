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

frame_count = 0

landmarks = []

REQUESTED_WIDTH = 300

width                 = 0
height                = 0
ratio                 = 0
upsample_ratio        = 0
new_width             = 0
new_height            = 0
has_calculated_ratios = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # TODO only run this for the first frame, assuming frame size never changes
    if not has_calculated_ratios:
        width                 = frame.shape[1]
        height                = frame.shape[0]
        ratio                 = REQUESTED_WIDTH / float(width)
        upsample_ratio        = float(width)    / REQUESTED_WIDTH
        new_width             = int(round(width  * ratio))
        new_height            = int(round(height * ratio))
        has_calculated_ratios = True

    frame_count += 1

    # Only do detection every other frame
    if frame_count >= 2:

        resized_frame = cv.resize(frame, (new_width, new_height), interpolation = cv.INTER_AREA)

        detected_faces = detector(resized_frame, 1)

        for face in detected_faces:
            shape     = predictor(resized_frame, face)
            landmarks = map(lambda idx: shape.part(idx) , facial_landmark_indices.values())

        frame_count = 0

    for landmark in landmarks:
        pos = (
            int(round(landmark.x * upsample_ratio)),
            int(round(landmark.y * upsample_ratio)),
        )
        cv.circle(frame, pos, 3, (0, 255, 0), -1)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
