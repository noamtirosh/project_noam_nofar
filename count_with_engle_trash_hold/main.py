import cv2
import mediapipe as mp
import numpy as np
from get_feature_vec import get_angles , get_angular_volcity ,get_min_visability ,choose_side
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
visibility_threshold = 98
up_threshold = 170
down_threshold = 140
n_sequence_direction = 4
up_direction = np.array([1])
down_direction = np.array([-1])

def calculate_2d_angles(landmarks, joint_list):
        # Loop through joint sets
    angles = []
    for joint in joint_list:
        a = np.array([landmarks[joint[0]].x, landmarks[joint[0]].y])  # First coord
        b = np.array([landmarks[joint[1]].x, landmarks[joint[1]].y])  # Second coord
        c = np.array([landmarks[joint[2]].x, landmarks[joint[2]].y]) # Third coord
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        angles.append(angle)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    v_1 = c-b
    v_2 = a-b
    cos_teta = np.inner(v_1,v_2)/(np.linalg.norm(v_1)*np.linalg.norm(v_2))
    rad = np.arccos(cos_teta)
    #radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_2d_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def check_visible(landmark,indexes):
    return sum(landmark[indexes].visibility)


def choose_position_by_v(velacity_vector:np.ndarray,direction_vector:np.ndarray,threshold_vector:np.ndarray,last_frames_arr:np.ndarray):
    if np.abs(velacity_vector)>threshold_vector:
        add_to_frames = velacity_vector*direction_vector
    else:
        add_to_frames = 0
    for ind ,frame in enumerate(last_frames_arr[1:]):
        last_frames_arr[ind] = frame
    last_frames_arr[ind+1] = add_to_frames
    return not np.any(last_frames_arr <= 0)


video_path = r"C:\Users\noam\Downloads\VID_20211128_163923 (1).mp4"
cap =cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(r"C:\Users\noam\Downloads\VID_20211128_164022.mp4")

# Curl counter variables
counter = 0
stage = None
last_frame_angles = 0
last_frame_time = 0
right_side = False
last_frames_arr = np.zeros(n_sequence_direction)
v_angle_direction =up_direction
v_angle_threshold =np.array([1])
stage = "down"
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
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
            if counter == 0:
                right_side = choose_side(landmarks)
            angels = get_angles(landmarks, right_side)
            if not last_frame_angles:
                last_frame_angles = angels
            else:
                volacity_in = np.array([last_frame_angles,angels])
                last_frame_angles = angels
                angle_v = get_angular_volcity(volacity_in,[0,1])
                if choose_position_by_v([angle_v[3]],v_angle_direction,v_angle_threshold,last_frames_arr):
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
            cv2.putText(image, str(angle_v[3]),
                        tuple(np.multiply([0.5,0.5], [640, 480]).astype(int)),
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
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)
        #print(hip[0:])
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



