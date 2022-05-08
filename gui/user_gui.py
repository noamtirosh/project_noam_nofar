import pygame
import json
from pygame.locals import *
import cv2
import mediapipe as mp
import torch
from genric_net.genric_net import load_network
import numpy as np
from tools.get_feature_vec import get_angle_vec_input, get_min_visability, get_vec_input ,choose_side


def GetInput():
    key = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE: quit()


pygame.init()
pygame.mixer.init()
open_message = pygame.mixer.Sound(r"C:\git_repos\project_noam_nofar\gui\open_message.mp3")
start_mes = pygame.mixer.Sound(r"C:\git_repos\project_noam_nofar\gui\start_pos.mp3")
song_2 = pygame.mixer.Sound(r"C:\git_repos\project_noam_nofar\tabata-timer-main\tabata-timer-main\front\media\rest.mp3")
# song_1.play(0)
# song_2.play(0)
#init media pip
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

Screen = max(pygame.display.list_modes())
icon = pygame.Surface((1,1)); icon.set_alpha(0); pygame.display.set_icon(icon)
pygame.display.set_caption("[Program] - [Author] - [Version] - [Date]")
Surface  = pygame.display.set_mode(Screen,FULLSCREEN)

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)
print(pygame.display.Info())
exercises_dict ={}
with open('exercise_plane.json') as json_file:
    data = json.load(json_file)
    exercises_dict = json.loads(data)

#init video
video_path = r"C:\Users\noam\Downloads\WhatsApp Video 2022-04-23 at 21.20.52.mp4"
cap = cv2.VideoCapture(1)
for exercise in exercises_dict['exercises']:
    count_model = load_network(exercise['count_model_path'])
    down_model = load_network(exercise['down_model_path'])
    open_message.play(0)
    counter = 0
    up_state = False
    visible_flag = False
    visibility_threshold = 0.80
    start_position = False
    start_position_threshold = 0.97
    time_var = pygame.time.get_ticks()
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
            except:
                visibility = 0
                continue
            if counter == 0:
                is_right_side = choose_side(landmarks)
                angels_visability = get_min_visability(landmarks, is_right_side)
                visible_flag = not np.any(angels_visability < visibility_threshold)
            if visible_flag:
                rec_color = (117, 245, 16)  # set rec color to green
                # use count_model to find start position
                model_input = get_angle_vec_input(landmarks)
                model_input = torch.Tensor(model_input)
                # Calculate the class probabilities (softmax) for img
                count_model.eval()
                with torch.no_grad():
                    output = count_model.forward(model_input.view(1, 56))
                ps = torch.exp(output)
                top_p, top_class_main = ps.topk(1, dim=1)
                if start_position:
                    if top_p < start_position_threshold:
                        # print("middle")
                        pass
                    else:
                        if top_class_main == 0:
                            if up_state:
                                up_state = False
                                counter = counter+1
                            down_input = get_vec_input(landmarks)
                            down_input = torch.Tensor(down_input)
                            with torch.no_grad():
                                output = down_model.forward(down_input.view(1, 32))
                            ps = torch.exp(output)
                            top_p, top_class = ps.topk(1, dim=1)
                            print(top_class)
                        else:
                            up_state = True
                else:
                    if not (top_class_main == 0 and top_p > start_position_threshold):
                        time_var = pygame.time.get_ticks()
                    if pygame.time.get_ticks() - time_var > 7*1000:
                        start_position = True
                        start_mes.play(0)


            else:
                rec_color = (117, 16, 245)  # set rec color to red

            cv2.rectangle(image, (0, 0), (225, 73), rec_color, -1)
            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(int(counter)),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow('Mediapipe Feed', image)
            imS = cv2.resize(image, (800, 600))  # Resize image
            impg = pygame.image.frombuffer(imS.tobytes(), imS.shape[1::-1],"BGR")
            Surface.blit(impg, (0, 0))
            pygame.display.update()

            # cv2.imshow('Mediapipe Feed', imS)

            GetInput()
# video_path = r"C:\Users\noam\Videos\project\currect\WIN_20211128_16_48_23_Pro.mp4"














# Opening JSON file

# exercises_dict = {
#     'exercises': [
#         {
#             'name': 'bridge',
#             'num_of_reps': 20,
#             'num_of_sets': 4,
#             'count_model_path': r"C:\git_repos\project_noam_nofar\models\work very good\model_angle_vec_20.4.pth",
#             'down_model_path': r"C:\git_repos\project_noam_nofar\models\down_model.pth",
#
#         }
#     ]
# }
# json_string = json.dumps(exercises_dict)
# with open('exercise_plane.json', 'w') as outfile:
#     json.dump(json_string, outfile)



while pygame.mixer.get_busy():
    pygame.time.delay(10)
