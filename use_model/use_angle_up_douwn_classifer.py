import cv2
import mediapipe as mp
import torch
from train_model.train_model_angle import Classifier
import numpy as np
from tools.get_feature_vec import get_angles, get_min_visability

model = Classifier()


#init model
model_path = r"C:\git_repos\project_noam_nofar\train_model\checkpoint_angle.pth"
checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['state_dict'])
model.load_state_dict(checkpoint)
#init media pip
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#init video
video_path = r"C:\Users\noam\Videos\project\VID_20220102_124819.mp4"
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if (type(frame) != np.ndarray):
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            model_input = []
            # left side
            angels_l = get_angles(landmarks, False)
            angels_visability_l = get_min_visability(landmarks, False)
            for i in range(len(angels_l)):
                model_input.append(angels_l[i] / 360)  # normlize
                model_input.append(angels_visability_l[i])
            angels = get_angles(landmarks, True)
            angels_visability = get_min_visability(landmarks, True)
            for i in range(len(angels)):
                model_input.append(angels[i] / 360)  # normlize
                model_input.append(angels_visability[i])

            model_input = torch.Tensor(model_input)
            # Calculate the class probabilities (softmax) for img
            model.eval()
            with torch.no_grad():
                output = model.forward(model_input.view(1, 24))
            ps = torch.exp(output)
            # cv2.imshow('Mediapipe Feed', image)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                             circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                             circle_radius=2)
                                      )
            imS = cv2.resize(image, (960, 540))  # Resize image
            cv2.imshow('Mediapipe Feed', imS)
        except:
            visibility = 0
            pass

        top_p, top_class = ps.topk(1, dim=1)
        print(ps)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
