import pygame
import json

from pygame.locals import *
import cv2
import mediapipe as mp
import torch
from genric_net.genric_net import load_network
import numpy as np
from tools.get_feature_vec import get_angle_vec_input, get_min_visability, get_vec_input, choose_side
import csv
import os
from datetime import datetime



# const
VIS_THRESHOLD = 0.80
MISTAKE_P_THRESHOLD = 0.7
STATE_POS_THRESHOLD = 0.97
P_OF_REP = 0.1
P_OF_SET = 0.2

RED_COLOR = (117, 16, 245)
GREEN_COLOR = (117, 245, 16)
BLACK_COLOR = (0, 0, 0)

START_POS = 0
END_POS = 1
MIDDLE_POS = 2

TIME_TO_CHECK_MISTAKES = 2000
TIME_IN_START_POS_TO_START = 4000




class UserGui:
    def __init__(self, exercise_json_path):
        # load json
        self.json_dict = {}
        with open(exercise_json_path) as json_file:
            data = json.load(json_file)
            self.json_dict = json.loads(data)
        self.csv_path = self.json_dict['plane_csv']
        self.exercises_list = self.json_dict['exercises']
        self.init_plane_csv()
        num_of_set = 0
        for exercise in self.exercises_list:
            num_of_set += exercise['num_of_sets']
        self.exercises_progress = np.zeros(num_of_set)
        self.total_set_ind = 0
        self.exercise_ind = 0
        self.init_pygame()
        self.init_media_pipe()

    def init_pygame(self):
        global open_message, start_mes, almost_done
        # init gui
        pygame.init()
        pygame.mixer.init()
        self.clock = pygame.time.Clock()
        self.start_mes = pygame.mixer.Sound(r"start_pos.mp3")
        almost_done = [r"good_job2.mp3", r"last_one.mp3", r"two_more.mp3", r"three_more.mp3"]
        Screen = max(pygame.display.list_modes())
        self.Surface = pygame.display.set_mode(Screen, FULLSCREEN)
        self.width = self.Surface.get_width()
        # stores the height of the screen into a variable
        self.height = self.Surface.get_height()
        self.button_width = int(self.width/5)
        self.button_height = int(self.height/5)
        icon = pygame.Surface((1, 1))
        icon.set_alpha(0)
        pygame.display.set_icon(icon)
        pygame.display.set_caption("[Program] - [Author] - [Version] - [Date]")
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        # print(pygame.display.Info())

    def init_media_pipe(self):
        # init video
        video_path = r"C:\Users\noam\Downloads\WhatsApp Video 2022-04-23 at 21.20.52.mp4"
        self.mp_pose = mp.solutions.pose
        self.cap = cv2.VideoCapture(video_path)
        self.mp_drawing = mp.solutions.drawing_utils

    def crate_json_file(self):
        pass

    def init_exercise(self, exercise_dict):
        # get exercise params from json
        self.count_model = load_network(exercise_dict['count_model_path'])
        self.start_pos_model = load_network(exercise_dict['down_model_path'])
        self.end_pos_model = load_network(exercise_dict['down_model_path'])
        self.middle_pos_model = load_network(exercise_dict['down_model_path'])
        open_message_path = exercise_dict['open_message']
        self.open_message = pygame.mixer.Sound(open_message_path)
        self.n_sets = exercise_dict['num_of_sets']
        self.n_reps = exercise_dict['num_of_reps']
        self.exercise_name = exercise_dict['name']
        self.break_between_sets = exercise_dict['break_time']
        self.time_var = pygame.time.get_ticks()
        self.user_message = r'stand before the camera'

    def check_visability(self):
        if not self.visible_flag:
            self.user_message = r'the camera cant see you clearly'
            self.rec_color = RED_COLOR  # set rec color to red
            is_right_side = choose_side(self.landmarks)  # choose the dominent visible side
            angels_visability = get_min_visability(self.landmarks, is_right_side)
            self.visible_flag = not np.any(angels_visability < VIS_THRESHOLD)

    def check_exe_start(self):
        if not self.is_exe_started:  # Wait for the user to get into position (10 sec)
            if not (self.pos_state == START_POS and self.state_p > STATE_POS_THRESHOLD):
                self.time_var = pygame.time.get_ticks()
                # reset mistakes arrays
                self.state_mistake_arr = np.zeros([3, 5])
            elif pygame.time.get_ticks() - self.time_var > TIME_TO_CHECK_MISTAKES:
                # TODO check for common error
                mistake_p, mistake_ind = self.use_error_model(START_POS)
                if mistake_p > MISTAKE_P_THRESHOLD:
                    self.state_mistake_arr[START_POS, mistake_ind] += 1

            if pygame.time.get_ticks() - self.time_var > TIME_IN_START_POS_TO_START:
                if np.any(self.state_mistake_arr[START_POS,:] > P_OF_REP):
                    self.time_var = pygame.time.get_ticks()
                    # TODO play error voice
                else:
                    self.is_exe_started = True
                    self.user_message = ''
                    self.start_mes.play(0)
                # reset mistakes arrays
                self.state_mistake_arr = np.zeros([3, 5])

    def update_gui(self):
        self.get_input()

        # cv2.imshow('Mediapipe Feed', image)



        imS = cv2.resize(self.frame, (self.width, int(self.height*4/5)))  # Resize image
        impg = pygame.image.frombuffer(imS.tobytes(), imS.shape[1::-1], "BGR")
        self.Surface.blit(impg, (0, int(self.height/5)))
        color_dark = (100, 100, 100)
        color = (255, 255, 255)

        # light shade of the button
        color_light = (170, 170, 170)

        smallfont = pygame.font.SysFont('Corbel', int(self.button_height/2))

        if self.user_message == '':
            # exercise name
            pygame.draw.rect(self.Surface, color, [0, 0, 2*self.button_width, self.button_height])
            text = smallfont.render(str(self.exercise_name), True, BLACK_COLOR)
            self.Surface.blit(text, (0, int(self.height/10)/2))
            # skip button
            pygame.draw.rect(self.Surface, RED_COLOR, [self.width - self.button_width, 0, self.button_width, self.button_height])
            text = smallfont.render("SKIP", True, color)
            self.Surface.blit(text, (self.width - self.button_width+int(self.button_height/2), int(self.button_height/4)))
            # set counter
            pygame.draw.rect(self.Surface, color_light, [2*self.button_width, 0, self.button_width, self.button_height])
            text = smallfont.render('SET', True, BLACK_COLOR)
            self.Surface.blit(text, (2*self.button_width, 0))
            text = smallfont.render(str(int(self.set_counter)) + '/' + str(int(self.n_sets)), True, color)
            self.Surface.blit(text, (2*self.button_width, int(self.button_height/2)))
            # rep counter
            pygame.draw.rect(self.Surface, self.rec_color, [3*self.button_width, 0, self.button_width, self.button_height])
            text = smallfont.render('REPS', True, BLACK_COLOR)
            self.Surface.blit(text, (3*self.button_width, 0))
            text = smallfont.render(str(int(self.rep_counter)) + '/' + str(int(self.n_reps)), True, color)
            self.Surface.blit(text, (3*self.button_width, int(self.height/10)))
        else:
            # message to user
            pygame.draw.rect(self.Surface, self.rec_color, [0, 0, self.width, self.button_height])
            text = smallfont.render(str(self.user_message), True, BLACK_COLOR)
            self.Surface.blit(text, (0, int(self.button_height/4)))

        pygame.display.update()
        # cv2.imshow('Mediapipe Feed', imS)


    def use_count_model(self):
        model_input = get_angle_vec_input(self.landmarks)
        model_input = torch.Tensor(model_input)
        # Calculate the class probabilities (softmax) for img
        top_p, top_class_main = 0, 0
        self.count_model.eval()
        with torch.no_grad():
            output = self.count_model.forward(model_input.view(1, 56))
            ps = torch.exp(output)
            top_p, top_class_main = ps.topk(1, dim=1)
        return top_p, top_class_main

    def use_error_model(self, state):
        error_input = get_vec_input(self.landmarks)  # TODO change if needed
        error_input = torch.Tensor(error_input)
        error_ind, error_p = 0, 0
        with torch.no_grad():
            if state == START_POS:
                error_output = self.start_pos_model.forward(error_input.view(1, 32))
            elif state == END_POS:
                error_output = self.end_pos_model.forward(error_input.view(1, 32))
            else:
                error_output = self.middle_pos_model.forward(error_input.view(1, 32))
            ps = torch.exp(error_output)
            error_p, error_ind = ps.topk(1, dim=1)
        return error_p, error_ind

    def update_counter(self):
        state = START_POS
        if self.state_p < STATE_POS_THRESHOLD:
            # Middle_position
            state = MIDDLE_POS
        else:
            if self.pos_state == START_POS:
                # start state
                state = START_POS
            else:
                # end position
                state = END_POS
                self.current_state = END_POS

        mistake_p, mistake_ind = self.use_error_model(state)
        if mistake_p > MISTAKE_P_THRESHOLD:
            self.state_mistake_arr[state, mistake_ind] += 1

        if state == START_POS and self.current_state == END_POS:
            self.rep_counter += 1
            # normalize arr
            self.state_mistake_arr = self.state_mistake_arr / self.state_mistake_arr.sum(axis=1).reshape([3, 1])
            # check mistake threshold and update mistake for set
            self.set_mistake_arr[self.state_mistake_arr > P_OF_REP] += 1
            # reset mistakes arrays
            self.state_mistake_arr = np.zeros([3, 5])
            self.current_state = START_POS

    def check_set(self):
        if self.rep_counter >= self.n_reps:
             self.end_set()
        if self.set_counter >= self.n_sets:
            self.is_exe_ended = True
            # TODO handle error if needed
            # take brake
            self.count_down()
            pass

    def run_exercise(self, exercise_ind):
        # reset exercise vars
        self.rep_counter = 0
        self.set_counter = 0
        self.visible_flag = False
        self.is_exe_started = False
        self.is_exe_ended = False
        self.visible_flag = False
        self.current_state = START_POS

        self.state_p, self.pos_state = 0, 0

        self.state_mistake_arr = np.zeros([3, 5])
        self.set_mistake_arr = np.zeros([3, 5])

        exercise_dict = self.exercises_list[exercise_ind]
        self.init_exercise(exercise_dict)
        self.open_message.play(0)
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                # Get the current image landmarks using MediaPipe.Pose
                ret, self.frame = self.cap.read()
                if (type(self.frame) != np.ndarray):
                    break
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                # Make detection
                results = pose.process(image)
                # Recolor back to BGR
                try:
                    self.landmarks = results.pose_landmarks.landmark
                except:
                    visibility = 0  # TODO check if needed
                    continue
                self.state_p, self.pos_state = self.use_count_model()

                if self.is_exe_started:
                    self.update_counter()
                    self.check_set()
                    # count down at the end
                    if self.rep_counter >= self.n_reps - 3:
                        pygame.mixer.Sound(almost_done[self.n_reps - self.rep_counter]).play(0)
                elif self.visible_flag:
                    self.rec_color = GREEN_COLOR
                    self.user_message = r'move to the start position'
                    self.check_exe_start()
                else:
                    self.check_visability()

                self.update_gui()

                # check if user is in place

    def get_num_of_exercise(self):
        return len(self.exercises_list)

    def count_down(self, time_sec):
        font = pygame.font.SysFont(None, 100)
        counter = time_sec
        text = font.render(str(counter), True, (0, 128, 0))
        timer_event = pygame.USEREVENT + 1
        pygame.time.set_timer(timer_event, 1000)
        run = True
        while run:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == timer_event:
                    text = font.render(str(counter), True, (0, 128, 0))
                    if counter == 0:
                        pygame.time.set_timer(timer_event, 0)
                        run = False
            self.Surface .fill((255, 255, 255))
            text_rect = text.get_rect(center=self.Surface .get_rect().center)
            self.Surface .blit(text, text_rect)
            pygame.display.flip()

    def end_gui(self):
        pygame.quit()
        # save csv file
        self.update_plane_csv()
        exit()

    def init_plane_csv(self):
        if not os.path.isfile(self.csv_path):
            first_row = ["date"]
            for exercise in self.exercises_list:
                for set_num in range(exercise['num_of_sets']):
                    first_row += ['{}-set-{}'.format(exercise['name'],set_num+1)]
            with open(self.csv_path, mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(first_row)

    def update_plane_csv(self):
        row = list(self.exercises_progress.flatten())
        time = datetime.now()
        date = time.strftime("%b %d, %Y")
        row.insert(0, date)
        with open(self.csv_path, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)

    def get_input(self):
        key = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE: quit()
            if event.type == pygame.QUIT:
                self.end_gui()
            # checks if a mouse is clicked
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.user_message == '':
                    mouse = pygame.mouse.get_pos()
                    if self.width - self.button_width <= mouse[0] <= self.width:
                        self.end_set()

    def end_set(self):
        self.set_counter += 1
        self.exercises_progress[self.total_set_ind] = self.rep_counter
        self.total_set_ind += 1
        self.rep_counter = 0
        # give feedback to user
        # normalize mistake_for_set
        self.set_mistake_arr = self.set_mistake_arr / self.n_reps
        # remove correct ind
        self.set_mistake_arr = self.set_mistake_arr[:, 1:]
        # check set threshold
        if np.any(self.set_mistake_arr > P_OF_SET):
            ind = np.argmax(self.set_mistake_arr)
        # TODO add error
        if self.break_between_sets:
            self.count_down(self.break_between_sets)








if __name__ == '__main__':
    print('hi')
    gui = UserGui(r"exercise_plan_18-5.json")
    for exercise_ind in range(gui.get_num_of_exercise()):
        gui.run_exercise(0)


#
#
#                             # exercise start
#                             while set_counter < n_sets:
#
#
# errors = [exercise['no_mistake'], exercise['error_type1'], exercise['error_type2'],
#           exercise['error_type3']]
# pygame.mixer.Sound(errors[up_miss]).play(0)
# sets = exercise['num_of_sets']
# if set_ind == sets:
#     pygame.time.delay(2500)
#     pygame.mixer.Sound(
#         r"C:\Users\nofar\PycharmProjects\pythonProject2\project_noam_nofar\gui\finished.mp3").play(
#         0)
# else:
#     pygame.time.delay(2500)
#     pygame.mixer.Sound(
#         r"C:\Users\nofar\PycharmProjects\pythonProject2\project_noam_nofar\gui\rest.mp3").play(
#         0)
# between_sets(mistake_reps_up, mistake_reps_middle, mistake_reps_down,
#              exercise, set_counter)  # Between set feedback
