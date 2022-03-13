import cv2
import mediapipe as mp
import numpy as np

# right side
left_joint_list = [[13, 15, 17], [11, 13, 15], [23, 11, 13], [25, 23, 11], [27, 25, 23],[31, 27, 25]]  # wrist elbow shoulder hip knee ankle
# left side
right_joint_list = np.array(left_joint_list) + 1


def get_angles(scelton_points: np.ndarray, right_side: bool):
    # Loop through joint sets
    angles = []
    if right_side:
        joint_list = right_joint_list
    else:
        joint_list = left_joint_list
    for joint in joint_list:
        a = np.array([scelton_points[joint[0]].x, scelton_points[joint[0]].y])  # First coord
        b = np.array([scelton_points[joint[1]].x, scelton_points[joint[1]].y])  # Second coord
        c = np.array([scelton_points[joint[2]].x, scelton_points[joint[2]].y])  # Third coord
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        # if angle > 180.0:
        #     angle = 360 - angle
        angles.append(angle)
    return angles


def get_min_visability(scelton_points: np.ndarray, right_side: bool):
    if right_side:
        joint_list = right_joint_list
    else:
        joint_list = left_joint_list
    visability_vector = []
    for joint in joint_list:
        angle_vis = [scelton_points[point].visibility for point in joint]
        visability_vector.append(min(angle_vis))
    return visability_vector


def choose_side(scelton_points: np.ndarray):
    right_vis = np.mean(get_min_visability(scelton_points, True))
    left_vis = np.mean(get_min_visability(scelton_points, False))
    return right_vis > left_vis


def get_angular_volcity(angels: np.ndarray, time):
    # Loop through joint sets
    angle_delta = angels[-1] - angels[0]
    return angle_delta / (time[-1] - time[-0])
