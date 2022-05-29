import cv2
import mediapipe as mp
import torch
from genric_net.genric_net import load_network
import numpy as np
from tools.get_feature_vec import  get_angle_vec_input, choose_side , count_model_input_with_side

RIGHT_SIDE = 0
LEFT_SIDE = 1

side_to_use = RIGHT_SIDE
flip_side_flag = False # right side
set_side_flag = False

#init model

model_path = r"C:\git_repos\project_noam_nofar\train_model\count_model_right_side.pth"
model = load_network(model_path)

#init media pip
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#init video
video_path = r"C:\Users\noam\Downloads\WhatsApp Video 2022-04-23 at 21.20.52.mp4"
# video_path = r"C:\Users\noam\Downloads\עותק של הרמה נכונה.mp4"

cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_origen:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_side:
        while cap.isOpened():
            ret, frame = cap.read()
            if (type(frame) != np.ndarray):
                break
            if flip_side_flag:
                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make detection
            # Recolor back to BGR

            try:
                # check if need to flip image
                if not set_side_flag:
                    results = pose_origen.process(image)
                    landmarks = results.pose_landmarks.landmark

                    if choose_side(landmarks) :
                        if side_to_use == LEFT_SIDE:
                            flip_side_flag = True
                    elif side_to_use == RIGHT_SIDE:
                        flip_side_flag = True
                    set_side_flag = True
                else:
                    results = pose_side.process(image)
                    landmarks = results.pose_landmarks.landmark

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                model_input = count_model_input_with_side(landmarks, side_to_use)
                model_input = torch.Tensor(model_input)
            except:
                continue
                        # Calculate the class probabilities (softmax) for img
            model.eval()
            with torch.no_grad():
                output = model.forward(model_input.view(1, 14))
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            print(ps)
            # cv2.imshow('Mediapipe Feed', image)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                             circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                             circle_radius=2)
                                      )
            imS = cv2.resize(image, (960, 540))  # Resize image
            cv2.imshow('Mediapipe Feed', imS)



            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
