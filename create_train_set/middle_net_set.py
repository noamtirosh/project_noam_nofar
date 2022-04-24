import cv2
import mediapipe as mp
import numpy as np
import glob

from tools.get_feature_vec import get_angle_input
from tools.save_to_csv import create_first_row, export_landmarks
from genric_net.genric_net import load_network
import torch

up_percentage = 0.95
down_percentage = 0.90


def create_vec_dir_data(video_list: list, csv_file_name: str, up_down_model_path: str):
    # for angle modle
    model = load_network(up_down_model_path)
    create_first_row(csv_file_name)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    for video in video_list:
        cap = cv2.VideoCapture(video)
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
                    model_input = get_angle_input(landmarks)
                    model_input = torch.Tensor(model_input)
                    # Calculate the class probabilities (softmax) for img
                    model.eval()
                    with torch.no_grad():
                        output = model.forward(model_input.view(1, 24))
                        ps = torch.exp(output)
                        print(ps)
                        if ps[0][1] < down_percentage and ps[0][0] < up_percentage:
                            export_landmarks(csv_file_name, landmarks, 'middle')
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

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    folder_to_use = r"C:\Users\noam\Videos\project\currect"
    video_list = glob.glob(folder_to_use + r"\*.mp4")
    model_path = r"C:\git_repos\project_noam_nofar\train_model\model_angle20.4.pth"
    create_vec_dir_data(video_list, folder_to_use + r"\middle_data.csv", model_path)
