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

landmarks = []

COLORS          = [ (255, 0, 0), (0, 255, 0), (0, 0, 255) ]
REQUESTED_WIDTH = 500

width                 = 0
height                = 0
ratio                 = 0
upsample_ratio        = 0
new_width             = 0
new_height            = 0
has_calculated_ratios = False

def get_color(idx):
    COLORS[face_idx % len(COLORS)]

def create_point(pos, color):
    return { "pos": pos, "color": color }

def create_line(p1, p2, color):
    return { "p1": p1, "p2": p2, "color": color }

def create_draw_buffer():
    return {
        "points": [],
        'lines':  [],
    }

def draw(frame, draw_buffer):
    for point in draw_buffer['points']:
        cv.circle(frame, point['pos'], 3, point['color'], -1)
    for line in draw_buffer['lines']:
        cv.line(frame, line['p1'], line['p2'], line['color'], 3)

    # Display the resulting frame
    frame = cv.flip(frame, 1)
    cv.imshow('frame', frame)

draw_buffer = create_draw_buffer()

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

    focal_length  = width
    center        = (width / 2, height / 2)
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

        draw_buffer = create_draw_buffer()

        # For each detected face
        for face_idx, face in enumerate(detected_faces):
            shape = predictor(resized_frame, face)

            # get each landmark we are interested in (we don't need all 68) add a circle to the frame
            def get_landmarks(landmark_idx):
                part = shape.part(landmark_idx)
                return (
                    int(round(part.x * upsample_ratio)),
                    int(round(part.y * upsample_ratio)),
                )

            landmark_positions = map(get_landmarks, FACIAL_LANDMARK_INDICES.values())
            print('landmark_positions', landmark_positions)

            # Do magic
            (success, rotation_vector, translation_vector) = cv.solvePnP(
                HEAD_MODEL_POINTS,
                np.array(landmark_positions, dtype="double"),
                camera_matrix,
                dist_coeffs,
                flags=cv.SOLVEPNP_ITERATIVE,
            )

            (nose_end_point2D) = cv.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]),
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeffs
            )

            color = get_color(face_idx)

            for pos in landmark_positions:
                point = create_point(pos, color)
                draw_buffer['points'].append(point)

            nose_tip_position = landmark_positions[0]

            p1 = (x, y) = nose_tip_position
            p2 = (int(round(nose_end_point2D[0][0][0])), int(round(nose_end_point2D[0][0][1])))
            line = create_line(p1, p2, color)

            draw_buffer['lines'].append(line)

    draw(frame, draw_buffer)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
