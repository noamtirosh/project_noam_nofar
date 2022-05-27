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
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_r:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_l:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if (type(frame) != np.ndarray):
                        break
                    # Make detection
                    # check if contain r\l for side
                    if Path(video).name.lower().count("right"):
                        right_side_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        left_side_image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                        right_side_image.flags.writeable = False
                        left_side_image.flags.writeable = False
                    elif Path(video).name.lower().count("left"):
                        left_side_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        right_side_image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                        right_side_image.flags.writeable = False
                        left_side_image.flags.writeable = False

                    else:
                        print("the file:{} not have side ".format(video))
                    # right_res = pose.process(right_side_image)
                    left_res = pose_l.process(left_side_image)
                    right_res = pose_r.process(right_side_image)
                    # Recolor back to BGR
                    right_side_image = cv2.cvtColor(right_side_image, cv2.COLOR_RGB2BGR)
                    right_side_image.flags.writeable = True

                    try:
                        right_landmarks = right_res.pose_landmarks.landmark
                        left_landmarks = left_res.pose_landmarks.landmark

                        # cv2.imshow('Mediapipe Feed', image)
                        mp_drawing.draw_landmarks(right_side_image, right_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                         circle_radius=2)
                                                  )
                        imS = cv2.resize(right_side_image, (960, 540))  # Resize image
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
    video_list =[r"C:\Users\noam\Downloads\right.mp4"]
    manual_create(video_list, folder_to_use +r"\data1.csv", False)
