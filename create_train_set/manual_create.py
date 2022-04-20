import cv2
import mediapipe as mp
import numpy as np
import glob

# from get_feature_vec import get_angles, get_angular_velocity, get_min_visability, choose_side
from tools.get_feature_vec import get_angles, get_angular_velocity, get_min_visability, choose_side
from tools.save_to_csv import create_first_row, export_landmarks


def manual_create(video_list: list, csv_file_name: str, run_video: bool = False):
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
                    # cv2.imshow('Mediapipe Feed', image)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                     circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                     circle_radius=2)
                                              )
                    imS = cv2.resize(image, (960, 540))  # Resize image
                    cv2.imshow('Mediapipe Feed', imS)
                    k = cv2.waitKey(10)
                    while k != ord('n'):
                        if k == ord('u'):
                            export_landmarks(csv_file_name, landmarks, 'up')
                            break
                        if k == ord('d'):
                            export_landmarks(csv_file_name, landmarks, 'douwn')
                            break
                        if k == ord('m'):
                            export_landmarks(csv_file_name, landmarks, 'middle')
                            break
                        if k == ord('q'):
                            exit()
                        if run_video:
                            break
                        k = cv2.waitKey(10)
                except:
                    visibility = 0
                    pass

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':

    folder_to_use = r"C:\Users\noam\Videos\project\currect"
    video_list = glob.glob(folder_to_use+r"\*.mp4")
    manual_create(video_list, folder_to_use +r"\data.csv", False)
