import numpy as np
from dataclasses import dataclass

# right side
left_joint_list = [[13, 15, 17], [11, 13, 15], [23, 11, 13], [25, 23, 11], [27, 25, 23],
                   [31, 27, 25]]  # wrist elbow shoulder hip knee ankle
# left side
right_joint_list = np.array(left_joint_list) + 1

left_tow_joints_list = [[7, 11], [11, 13], [13, 15], [15, 17], [23, 11], [25, 23], [27, 25], [31, 27]]
right_tow_joints_list = np.array(left_tow_joints_list) + 1


@dataclass
class SceltonPoint:
    """Class for keeping track of an item in inventory."""
    x: float
    y: float
    z: float
    visibility: float = 0


def get_angle_input(in_landmarks):
    model_input = []
    # left side
    angels_l = get_angles(in_landmarks, False)
    angels_visability_l = get_min_visability(in_landmarks, False)
    for i in range(len(angels_l)):
        model_input.append(angels_l[i] / 360)  # normlize
        model_input.append(angels_visability_l[i])
    angels = get_angles(in_landmarks, True)
    angels_visability = get_min_visability(in_landmarks, True)
    for i in range(len(angels)):
        model_input.append(angels[i] / 360)  # normlize
        model_input.append(angels_visability[i])

    return model_input


def get_vec_input(in_landmarks):
    # left side
    model_input = []
    for ind, tow_joints in enumerate(left_tow_joints_list):
        a = np.array([in_landmarks[tow_joints[0]].x, in_landmarks[tow_joints[0]].y])
        b = np.array([in_landmarks[tow_joints[1]].x, in_landmarks[tow_joints[1]].y])
        model_input.append(culc_vec_direction(a, b) / 360)
        model_input.append(min(in_landmarks[tow_joints[0]].visibility,
                                  in_landmarks[tow_joints[1]].visibility))

    # right side
    for ind, tow_joints in enumerate(right_tow_joints_list):
        a = np.array([in_landmarks[tow_joints[0]].x, in_landmarks[tow_joints[0]].y])
        b = np.array([in_landmarks[tow_joints[1]].x, in_landmarks[tow_joints[1]].y])
        model_input.append(culc_vec_direction(a, b) / 360)
        model_input.append(min(in_landmarks[tow_joints[0]].visibility,
                                  in_landmarks[tow_joints[1]].visibility))
    return model_input


def get_angle_vec_input(in_landmarks):
    model_input = get_angle_input(in_landmarks)
    vec_input = get_vec_input(in_landmarks)
    model_input.extend(vec_input)
    return model_input


def get_angles(scelton_points: np.ndarray, right_side: bool):
    """
    use the coordinates (x,y) of the
    :param scelton_points:
    :param right_side:
    :return:
    """
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
        angle = culc_angle(a, b, c)
        angles.append(angle)
    return angles


def culc_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if type(angle) == np.float64:
        angle = np.array(angle)
    # choose higer angle
    mask = (a[1] + ((c - a)[1] / (c - a)[0]) * (b[0] - a[0])) > b[1]
    blunt_angle = angle[mask]
    blunt_angle[blunt_angle < 180] = 360 - blunt_angle[blunt_angle < 180]
    angle[angle > 180] = 360 - angle[angle > 180]
    angle[mask] = blunt_angle
    return angle


def culc_vec_direction(a: np.ndarray, b: np.ndarray, is_2d: bool = True):
    if is_2d:
        out_vec_dir = a[0:2] - b[0:2]

    else:
        out_vec_dir = a - b
    out_vec_dir = out_vec_dir / np.linalg.norm(out_vec_dir, axis=0)
    radians = np.arctan2(out_vec_dir[1], out_vec_dir[0])
    return np.degrees(radians) + 180.0


def get_min_visability(scelton_points: np.ndarray, right_side: bool):
    if right_side:
        joint_list = right_joint_list
    else:
        joint_list = left_joint_list
    visability_vector = []
    for joint in joint_list:
        angle_vis = [scelton_points[point].visibility for point in joint]
        visability_vector.append(min(angle_vis))
    return np.array(visability_vector)


def choose_side(scelton_points: np.ndarray):
    right_vis = np.mean(get_min_visability(scelton_points, True))
    left_vis = np.mean(get_min_visability(scelton_points, False))
    return right_vis > left_vis


def get_angular_velocity(angels: np.ndarray, time):
    """
    return velocity vector, take last frame from angels matrix and subtract the first frame from it
    and dived by time delta
    :param angels: matrix ech row is a new frame
    :param time:vector of time that ech frame taken at
    :return:
    """
    # Loop through joint sets
    angle_delta = angels[-1] - angels[0]
    return angle_delta / (time[-1] - time[-0])
