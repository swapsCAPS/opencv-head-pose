import numpy as np
import os
import cv2 as cv
import dlib
import time
import threading
import itertools

cap            = cv.VideoCapture(0)
detector       = dlib.get_frontal_face_detector()
predictor_path = "{}/Downloads/shape_predictor_68_face_landmarks.dat".format(os.environ['HOME'])
predictor      = dlib.shape_predictor(predictor_path)

FACIAL_LANDMARK_INDICES = {
    "eye_left":    45,
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

COLORS = [ (255, 0, 0), (0, 255, 0), (0, 0, 255) ]

def make_get_landmarks(resized_frame, upsample_ratio, colors, facial_landmark_indices):
    def get_landmarks((face_idx, face)):
        shape = predictor(resized_frame, face)

        # Reuse colors if too many faces
        if face_idx > len(colors) - 1:
            face_idx = face_idx - len(colors)

        return map(lambda landmark_idx:
            {
                "pos": (
                    int(round(shape.part(landmark_idx).x * upsample_ratio)),
                    int(round(shape.part(landmark_idx).y * upsample_ratio)),
                ),
                "color": colors[face_idx]
            }

        , facial_landmark_indices.values())
    return get_landmarks

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if not has_calculated_ratios:
        width                 = frame.shape[1]
        height                = frame.shape[0]
        ratio                 = REQUESTED_WIDTH / float(width)
        upsample_ratio        = float(width)    / REQUESTED_WIDTH
        new_width             = int(round(width  * ratio))
        new_height            = int(round(height * ratio))
        has_calculated_ratios = True

    frame_count += 1

    # Only do detection every n frame
    if frame_count >= 2:
        frame_count = 0

        resized_frame  = cv.resize(frame, (new_width, new_height), interpolation = cv.INTER_AREA)

        detected_faces = detector(resized_frame, 1)

        get_landmarks  = make_get_landmarks(
            resized_frame,
            upsample_ratio,
            facial_landmark_indices = FACIAL_LANDMARK_INDICES,
            colors                  = COLORS,
        )

        landmarks = map(get_landmarks, enumerate(detected_faces))
        landmarks = list(itertools.chain(*landmarks))

    for landmark in landmarks:
        cv.circle(frame, landmark['pos'], 3, landmark['color'], -1)

    # Display the resulting frame
    cv.flip(frame, 1)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
