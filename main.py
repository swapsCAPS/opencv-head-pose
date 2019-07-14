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
predictor_path = '{}/Downloads/shape_predictor_68_face_landmarks.dat'.format(os.environ['HOME'])
predictor      = dlib.shape_predictor(predictor_path)

DLIB_FRAME_WIDTH    = 450
RUN_EVERY_NTH_FRAME = 3

# From camera perspective (subject's left is our right...)
FACIAL_LANDMARK_INDICES = collections.OrderedDict([
    ('nose_tip',    30),
    ('chin',         8),
    ('eye_left',    36),
    ('eye_right',   45),
    ('mouth_left',  48),
    ('mouth_right', 54),
])

HEAD_MODEL_POINTS = collections.OrderedDict([
    #                 width, height, depth
    ('nose_tip',    (   0.0,    0.0,    0.0)),
    ('chin',        (   0.0, -330.0,  -65.0)),
    ('eye_left',    (-225.0,  170.0, -135.0)),
    ('eye_right',   ( 225.0,  170.0, -135.0)),
    ('mouth_left',  (-150.0, -150.0, -125.0)),
    ('mouth_right', ( 150.0, -150.0, -125.0)),
])


if not cap.isOpened():
    print('Cannot open camera')
    exit()

#  cap.set(3, 1280)
#  cap.set(4, 720)

def get_color(idx):
    colors = [
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
    return colors[idx % len(colors)]

def create_point(pos, color):
    return { 'pos': pos, 'color': color }

def create_line(p1, p2, color):
    return { 'p1': p1, 'p2': p2, 'color': color }

def create_draw_buffer():
    return {
        'points': [],
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

# The draw buffer will hold what to render during skipped detection frames
draw_buffer = create_draw_buffer()

# Place holder for all our frame info so we won't have to calc every frame
frame_info = {
    'has_calculated_frame_info': False,
}

frame_count = 0

while True:
    # Get teh frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Pressing q will break the loop
    if cv.waitKey(1) == ord('q'):
        break

    # Calculate frame info and stuff on the first frame
    if not frame_info['has_calculated_frame_info']:
        width                               = float(frame.shape[1])
        height                              = float(frame.shape[0])
        ratio                               = DLIB_FRAME_WIDTH / width
        frame_info['reverse_ratio']         = width / DLIB_FRAME_WIDTH
        frame_info['new_width']             = int(round(width  * ratio))
        frame_info['new_height']            = int(round(height * ratio))
        frame_info['focal_length']          = width
        frame_info['dist_coeffs']           = np.zeros((4,1)) # Assuming no lens distortion
        frame_info['camera_matrix']         = np.array([
            [frame_info['focal_length'], 0,                          width / 2],
            [0,                          frame_info['focal_length'], height / 2],
            [0,                          0,                          1]
        ], dtype = 'double')
        frame_info['has_calculated_frame_info'] = True

    # Only do detection every n frame
    frame_count += 1
    if frame_count >= RUN_EVERY_NTH_FRAME:
        frame_count = 0

        draw_buffer = create_draw_buffer()

        resized_frame  = cv.resize(frame, (frame_info['new_width'], frame_info['new_height']), interpolation = cv.INTER_AREA)

        detected_faces = detector(resized_frame, 1)

        if len(detected_faces) < 1:
            draw(frame, draw_buffer)
            continue

        # For each detected face
        for face_idx, face in enumerate(detected_faces):
            shape = predictor(resized_frame, face)

            # get each landmark we are interested in (we don't need all 68) add a circle to the frame
            def get_landmarks(landmark_idx):
                part = shape.part(landmark_idx)
                return (
                    int(round(part.x * frame_info['reverse_ratio'])),
                    int(round(part.y * frame_info['reverse_ratio'])),
                )

            landmark_positions = map(get_landmarks, FACIAL_LANDMARK_INDICES.values())

            # Do camera perspective magic by matching our facial landmarks to a predefined 3D model's landsmarks
            (success, rotation_vector, translation_vector) = cv.solvePnP(
                np.array(HEAD_MODEL_POINTS.values()),
                np.array(landmark_positions, dtype='double'),
                frame_info['camera_matrix'],
                frame_info['dist_coeffs'],
                flags=cv.SOLVEPNP_ITERATIVE,
            )

            # Do camera projection magic
            (nose_end_point2D, jacobian) = cv.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]),
                rotation_vector,
                translation_vector,
                frame_info['camera_matrix'],
                frame_info['dist_coeffs']
            )

            # Fill our draw buffer
            for pos in landmark_positions:
                point = create_point(pos, get_color(face_idx))
                draw_buffer['points'].append(point)

            p1 = (x, y) = landmark_positions[0]
            p2 = (int(round(nose_end_point2D[0][0][0])), int(round(nose_end_point2D[0][0][1])))
            line = create_line(p1, p2, get_color(face_idx + 2))

            draw_buffer['lines'].append(line)

        # Draw!
        draw(frame, draw_buffer)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
