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

COLORS = [
    (180,119,31),
    (14,127,255),
    (44,160,44),
    (40,39,214),
    (189,103,148),
    (75,86,140),
    (194,119,227),
    (127,127,127),
    (34,189,188),
    (207,190,23),
]
REQUESTED_WIDTH = 300

def get_color(idx):
    return COLORS[idx % len(COLORS)]

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

frame_info = {
    "width":                     0,
    "height":                    0,
    "ratio":                     0,
    "upsample_ratio":            0,
    "new_width":                 0,
    "new_height":                0,
    "has_calculated_frame_info": False,
}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if not frame_info['has_calculated_frame_info']:
        frame_info["width"]                 = frame.shape[1]
        frame_info["height"]                = frame.shape[0]
        frame_info["ratio"]                 = REQUESTED_WIDTH / float(frame['width'])
        frame_info["upsample_ratio"]        = float(frame['width'])    / REQUESTED_WIDTH
        frame_info["new_width"]             = int(round(frame['width']  * ratio))
        frame_info["new_height"]            = int(round(frame['height'] * ratio))
        frame_info["has_calculated_ratios"] = True
        frame_info["focal_length"]          = frame['width']
        frame_info["center"]                = (frame['width'] / 2, frame_info['height'] / 2)
        frame_info["camera_matrix"]         = np.array([
            [frame_info['focal_length'], 0,                          frame_info['center'][0]],
            [0,                          frame_info['focal_length'], frame_info['center'][1]],
            [0,                          0,                          1]
         ], dtype = "double")
        frame_info["dist_coeffs"] = np.zeros((4,1)) # Assuming no lens distortion

    frame_count += 1


    # Only do detection every n frame
    if frame_count >= 3:
        frame_count = 0

        resized_frame  = cv.resize(frame, (new_width, new_height), interpolation = cv.INTER_AREA)

        detected_faces = detector(resized_frame, 1)

        if len(detected_faces) > 0:

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

                # Do magic
                (success, rotation_vector, translation_vector) = cv.solvePnP(
                    HEAD_MODEL_POINTS,
                    np.array(landmark_positions, dtype="double"),
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

                for pos in landmark_positions:
                    point = create_point(pos, get_color(face_idx))
                    draw_buffer['points'].append(point)

                p1 = (x, y) = landmark_positions[0]
                p2 = (int(round(nose_end_point2D[0][0][0])), int(round(nose_end_point2D[0][0][1])))
                line = create_line(p1, p2, get_color(face_idx + 2))

                draw_buffer['lines'].append(line)

    draw(frame, draw_buffer)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
