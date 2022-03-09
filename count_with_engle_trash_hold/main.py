import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
visibility_threshold = 98
up_threshold = 170
down_threshold = 140


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



#video_path = r""
#cap =cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(r"C:\Users\noam\Downloads\VID_20211128_164022.mp4")

# Curl counter variables
counter = 0
stage = None

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

            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP .value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]

            visibility = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP .value].visibility + \
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE .value].visibility + \
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER .value].visibility)*100/3
            # Calculate angle
            angle = calculate_2d_angle(knee, hip, shoulder)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(hip[0:2], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if visibility > visibility_threshold:
                rec_color = (117, 245, 16)
                if angle > up_threshold:
                    stage = "up"
                if angle < down_threshold and stage == 'up':
                    stage = "down"
                    counter += 1
                    print(counter)

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

