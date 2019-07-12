import numpy as np
import os
import cv2 as cv
import dlib
import time
import threading
import itertools
import collections

cap            = cv.VideoCapture(0)
detector       = dlib.get_frontal_face_detector()
predictor_path = "{}/Downloads/shape_predictor_68_face_landmarks.dat".format(os.environ['HOME'])
predictor      = dlib.shape_predictor(predictor_path)

FACIAL_LANDMARK_INDICES = collections.OrderedDict([
    ("nose_tip",    30),
    ("chin",        8),
    ("eye_left",    36),
    ("eye_right",   45),
    ("mouth_left",  48),
    ("mouth_right", 54),
])

HEAD_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # Nose tip
    (0.0,    -330.0, -65.0),  # Chin
    (-225.0, 170.0,  -135.0), # Left eye left corner
    (225.0,  170.0,  -135.0), # Right eye right corne
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0,  -150.0, -125.0)  # Right mouth corner
])


if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0

p1 = ()
p2 = ()
landmarks = []

REQUESTED_WIDTH = 600

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

        return map(lambda (name, landmark_idx):
            {
                "name": name,
                "pos": (
                    int(round(shape.part(landmark_idx).x * upsample_ratio)),
                    int(round(shape.part(landmark_idx).y * upsample_ratio)),
                ),
                "color": colors[face_idx]
            }

        , facial_landmark_indices.items())
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

    size = frame.shape
    focal_length  = size[1]
    center        = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1]
     ], dtype = "double")

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    # Only do detection every n frame
    if frame_count >= 1:
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

        for landmark in landmarks:
            image_points = np.array(map(lambda l: l['pos'], landmark), dtype="double")
            (success, rotation_vector, translation_vector) = cv.solvePnP(
                HEAD_MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv.SOLVEPNP_ITERATIVE,
            )

            (nose_end_point2D, jacobian) = cv.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]),
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeffs
            )

            p1 = ( int(round(image_points[0][0])), int(round(image_points[0][1])))
            p2 = ( int(round(nose_end_point2D[0][0][0])), int(round(nose_end_point2D[0][0][1])))

        landmarks = list(itertools.chain(*landmarks))

    for landmark in landmarks:
        cv.circle(frame, landmark['pos'], 3, landmark['color'], -1)

    if len(landmarks) > 0:
        cv.line(frame, p1, p2, (0,255,0), 3)

    # Display the resulting frame
    frame = cv.flip(frame, 1)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
