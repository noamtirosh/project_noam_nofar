import cv2
import mediapipe as mp
import torch
from train_model.conv_media_pipe_example import Classifier
import numpy as np
from tools.get_mediapipe_example_vec import FullBodyPoseEmbedder
from genric_net.genric_net import load_network
from tools.get_feature_vec import get_cross_vec_input, get_conv_input
import pandas as pd
# import plotly.graph_objects as go

media_pip_input = FullBodyPoseEmbedder()

# init model
model_path = r"C:\git_repos\project_noam_nofar\train_model\middle_test.pth"
# model_path = r"C:\git_repos\project_noam_nofar\train_model\mp_example_wight.pth"

model = load_network(model_path)

# init media pip
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# init video
video_path = r"C:\Users\noam\Downloads\WhatsApp Video 2022-04-23 at 21.20.44.mp4"
# video_path = r"C:\Users\noam\Downloads\WhatsApp Video 2022-04-23 at 21.20.52.mp4"

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
            # model_input = get_cross_vec_input(landmarks)
            model_input = get_conv_input(landmarks)

            # landmarks_arr = np.zeros([33, 3])
            # for i in range(33):
            #     landmarks_arr[i, 0] = landmarks[i].x
            #     landmarks_arr[i, 1] = landmarks[i].y
            #     landmarks_arr[i, 2] = landmarks[i].z
            # model_input = media_pip_input(landmarks_arr)[:, 0:2]
            model_input = torch.Tensor(model_input)
            # Calculate the class probabilities (softmax) for img
            model.eval()
            with torch.no_grad():
                output = model.forward(model_input.view(1, 66))
            ps = torch.exp(output)
            # cv2.imshow('Mediapipe Feed', image)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                             circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                             circle_radius=2)
                                      )
            # mp_drawing.plot_landmarks(results.pose_world_landmarks,mp_pose.POSE_CONNECTIONS)
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


_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


def plot_landmarks(
    landmark_list,
    connections=None,
):
    if not landmark_list:
        return
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        out_cn = []
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        cn2 = {"xs": [], "ys": [], "zs": []}
        for pair in out_cn:
            for k in pair.keys():
                cn2[k].append(pair[k][0])
                cn2[k].append(pair[k][1])
                cn2[k].append(None)

    df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df["lm"] = df.index.map(lambda s: mp_pose.PoseLandmark(s).name).values
    fig = (
        px.scatter_3d(df, x="z", y="x", z="y", hover_name="lm")
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
    fig.add_traces(
        [
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )

    return fig