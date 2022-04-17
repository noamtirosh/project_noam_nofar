import cv2
import mediapipe as mp
import numpy as np
#from get_feature_vec import get_angles, get_angular_velocity, get_min_visability, choose_side
from tools.get_feature_vec import get_angles, get_angular_velocity, get_min_visability, choose_side

from save_to_csv import create_first_row, export_landmarks

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
visibility_threshold = 0.80

n_sequence_direction = 4
num_frames_for_velocity = 2
up_direction = np.array([1, -1])
down_direction = np.array([-1, 1])
joint_to_use = [3, 4]


def choose_position_by_v(velacity_vector: np.ndarray, direction_vector: np.ndarray, threshold_vector: np.ndarray,
                         last_frames_arr: np.ndarray):
    if not np.any(np.abs(velacity_vector) < threshold_vector):
        if np.any((velacity_vector * direction_vector) < 0):
            add_to_frames = -1
        else:
            add_to_frames = 1
    else:
        add_to_frames = 0
    for ind, frame in enumerate(last_frames_arr[1:]):
        last_frames_arr[ind] = frame
    last_frames_arr[ind + 1] = add_to_frames
    return not np.any(last_frames_arr <= 0)


"""
for using camera use VideoCapture(0)
"""
# cap = cv2.VideoCapture(0)

"""
for using video from the computer use cv2.VideoCapture(video_path)
"""

# video_path = r"C:\Users\noam\Downloads\VID_20211128_163923 (1).mp4"
# video_path = r"C:\Users\noam\Downloads\WhatsApp Video 2022-03-20 at 09.12.16.mp4"
video_path = r"C:\Users\noam\Downloads\VID_20211128_164022.mp4"
# video_path = r"C:\Users\noam\Downloads\VID_20211128_163923 (2).mp4"
# video_path = r"C:\Users\noam\Downloads\הרמה_תקינה (1).mp4"
# video_path = r"C:\Users\noam\Downloads\אגן_לא_סימטרי (2).mp4"
cap = cv2.VideoCapture(video_path)

"""
global variables for the counting
"""
counter = 0
stage = None
last_frame_angles = 0
last_frame_time = 0
is_right_side = False

last_frames_arr = np.zeros(n_sequence_direction)
v_angle_direction = up_direction
v_angle_threshold = np.array([1, 0.2])

stage = "down"

create_first_row()
# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        rec_color = (117, 16, 245)
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            export_landmarks(landmarks, stage)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # cv2.imshow('Mediapipe Feed', image)
            if counter == 0:
                is_right_side = choose_side(landmarks)
            angels = get_angles(landmarks, is_right_side)
            angels_visability = get_min_visability(landmarks, is_right_side)
            if not last_frame_angles:
                last_frame_angles = angels
            else:
                # create matrix for velocity
                angle_matrix = np.array([last_frame_angles, angels])
                last_frame_angles = angels

                time_vector = [0, 1]
                angular_velocity = get_angular_velocity(angle_matrix, time_vector)

                if np.any(angels_visability[joint_to_use] < visibility_threshold):
                    rec_color = (117, 16, 245)  # set rec color to red
                # stage = "not_visible"
                else:
                    rec_color = (117, 245, 16)  # set rec color to red
                    if choose_position_by_v([angular_velocity[joint_to_use]], v_angle_direction, v_angle_threshold,
                                            last_frames_arr):
                        if stage == "down":
                            v_angle_direction = down_direction
                            last_frames_arr = np.zeros(n_sequence_direction)
                            stage = "up"
                        elif stage == "up":
                            v_angle_direction = up_direction
                            stage = "down"
                            counter += 1
                            last_frames_arr = np.zeros(n_sequence_direction)
                            print(counter)

            # Visualize angle
            cv2.putText(image, str(angels[3]),
                        tuple(
                            np.multiply([results.pose_landmarks.landmark[24].x, results.pose_landmarks.landmark[24].y],
                                        [640, 360]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
        except:
            visibility = 0
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), rec_color, -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(int(counter)),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        cv2.imshow('Mediapipe Feed', image, )
        # print(hip[0:])
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
