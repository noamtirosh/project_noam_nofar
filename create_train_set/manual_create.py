import cv2
import mediapipe as mp
import numpy as np
import glob
from pathlib import Path

# from get_feature_vec import get_angles, get_angular_velocity, get_min_visability, choose_side
from tools.get_feature_vec import get_angles, get_angular_velocity, get_min_visability, choose_side
from tools.save_to_csv import create_first_row, export_landmarks


def manual_create(video_list: list, csv_file_name: str, run_video: bool = False):
    right_csv_name = csv_file_name.replace('.csv', '_right.csv')
    left_csv_name = csv_file_name.replace('.csv', '_left.csv')

    create_first_row(right_csv_name)
    create_first_row(left_csv_name)

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
                # check if contain r\l for side
                if Path(video).name.lower().count("r"):
                    right_side = image
                    left_side = cv2.flip(image, 1)
                elif Path(video).name.lower().count("l"):
                    left_side = image
                    right_side = cv2.flip(image, 1)
                else:
                    print("the file:{} not have side ".format(video))
                right_res = pose.process(right_side)
                left_res = pose.process(left_side)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(right_side, cv2.COLOR_RGB2BGR)
                try:
                    right_landmarks = right_res.pose_landmarks.landmark
                    left_landmarks = left_res.pose_landmarks.landmark

                    # cv2.imshow('Mediapipe Feed', image)
                    mp_drawing.draw_landmarks(image, right_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
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
                            export_landmarks(right_csv_name, right_landmarks, 'up')
                            export_landmarks(left_csv_name, left_landmarks, 'up')
                            break
                        if k == ord('d'):
                            export_landmarks(right_csv_name, right_landmarks, 'douwn')
                            export_landmarks(left_csv_name, left_landmarks, 'douwn')
                            break
                        if k == ord('m'):
                            export_landmarks(right_csv_name, right_landmarks, 'middle')
                            export_landmarks(left_csv_name, left_landmarks, 'middle')
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
    manual_create(video_list, folder_to_use +r"\data1.csv", False)
