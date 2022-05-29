import numpy as np
from dataclasses import dataclass

RIGHT_SIDE = 0
LEFT_SIDE = 1
FRONT_SIDE = 2

# right side
left_joint_list = [[13, 15, 17], [11, 13, 15], [23, 11, 13], [25, 23, 11], [27, 25, 23],
                   [31, 27, 25]]  # wrist elbow shoulder hip knee ankle
# left side
right_joint_list = np.array(left_joint_list) + 1
front_joint_list = []
front_joint_list.extend(right_joint_list)
front_joint_list.extend(left_joint_list)

left_tow_joints_list = [[7, 11], [11, 13], [13, 15], [15, 17], [23, 11], [25, 23], [27, 25], [31, 27]]
right_tow_joints_list = np.array(left_tow_joints_list) + 1
front_tow_joints_list = []
front_tow_joints_list.extend(right_tow_joints_list)
front_tow_joints_list.extend(left_tow_joints_list)


cross_body_joints_list = [[9, 10], [11, 12], [23, 24], [25, 26], [27, 28]]

# right_list, left_list = [], []
# right_list.append(0)
# for joint in self.mp_pose.PoseLandmark:
#     if joint._name_.lower().count('right'):
#         right_list.append(joint.value)
#     else:
#         left_list.append(joint.value)
left_point_list = [0, 1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
right_point_list = [0, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

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
    angels_l = get_angles_by_side(in_landmarks, False)
    angels_visability_l = get_min_visability(in_landmarks, False)
    for i in range(len(angels_l)):
        model_input.append(angels_l[i] / 360)  # normlize
        model_input.append(angels_visability_l[i])
    angels = get_angles_by_side(in_landmarks, True)
    angels_visability = get_min_visability(in_landmarks, True)
    for i in range(len(angels)):
        model_input.append(angels[i] / 360)  # normlize
        model_input.append(angels_visability[i])

    return model_input


def get_cross_vec_input(in_landmarks):
    model_input = get_vec_input(in_landmarks)
    for ind, tow_joint_cross in enumerate(cross_body_joints_list):
        a = np.array([in_landmarks[tow_joint_cross[0]].x, in_landmarks[tow_joint_cross[0]].y])
        b = np.array([in_landmarks[tow_joint_cross[1]].x, in_landmarks[tow_joint_cross[1]].y])
        model_input.append(culc_vec_direction(a, b) / 360)
        model_input.append(min(in_landmarks[tow_joint_cross[0]].visibility,
                               in_landmarks[tow_joint_cross[1]].visibility))
    return model_input


def get_vec_input(in_landmarks):
    # left side
    model_input = get_vec_input_by_side(in_landmarks, False)
    model_input.extend(get_vec_input_by_side(in_landmarks, True))
    return model_input


def get_vec_input_by_side(in_landmarks, right_side):
    joint_lists = [right_tow_joints_list, left_tow_joints_list]
    model_input = []
    for ind, tow_joints in enumerate(joint_lists[right_side]):
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


def get_angles_by_side(scelton_points: np.ndarray, side: bool):
    """
    use the coordinates (x,y) of the
    :param scelton_points:
    :param right_side:
    :return:
    """
    # Loop through joint sets
    angles = []
    joint_lists = [right_joint_list,left_joint_list]
    joint_list = joint_lists[side]
    for joint in joint_list:
        a = np.array([scelton_points[joint[0]].x, scelton_points[joint[0]].y])  # First coord
        b = np.array([scelton_points[joint[1]].x, scelton_points[joint[1]].y])  # Second coord
        c = np.array([scelton_points[joint[2]].x, scelton_points[joint[2]].y])  # Third coord
        angle = culc_angle(a, b, c)
        angles.append(angle)
    return angles


def get_angles(scelton_points: np.ndarray):
    angles = get_angles_by_side(scelton_points, True)
    angles.extend(get_angles_by_side(scelton_points, False))
    return angles


def get_vectors_with_side(in_landmarks, side):
    joint_lists = [right_tow_joints_list, left_tow_joints_list]
    model_input = []
    for ind, tow_joints in enumerate(joint_lists[side]):
        a = np.array([in_landmarks[tow_joints[0]].x, in_landmarks[tow_joints[0]].y])
        b = np.array([in_landmarks[tow_joints[1]].x, in_landmarks[tow_joints[1]].y])
        model_input.append(culc_vec_direction(a, b) / 360)
    return model_input


def get_vectors(in_landmarks):
    # left side
    model_input = get_vectors(in_landmarks, False)
    model_input.extend(get_vectors(in_landmarks, True))
    return model_input

# model_inputs func


def count_model_input_with_side(in_landmarks, side):
    model_input =[]
    angle_input = get_angles_by_side(in_landmarks, side)
    for i in range(len(angle_input)):
        model_input.append(angle_input[i] / 360)
    model_input.extend(get_vectors_with_side(in_landmarks, side))
    return model_input


def count_model_input(in_landmarks):
    model_input = get_angles(in_landmarks)
    model_input.extend(get_vectors(in_landmarks))
    return model_input


def get_point_loc_input(in_landmarks):
    model_input = []
    for joint in in_landmarks:
        model_input.append(joint.x)
        model_input.append(joint.y)
        # model_input.append(joint.z)
    return model_input


def get_point_loc_input_by_side(in_landmarks, side):
    model_input = []
    if side == LEFT_SIDE:
        landmarks_list = left_point_list
    elif side == RIGHT_SIDE:
        landmarks_list = right_point_list
    else:
        landmarks_list = [i for i in range(len(in_landmarks)+1)]
    for joint_ind, joint in enumerate(in_landmarks):
        if joint_ind in landmarks_list:
            model_input.append(joint.x)
            model_input.append(joint.y)
        # model_input.append(joint.z)
    return model_input

def get_conv_input(in_landmarks):
    model_input = []
    for ind, joint in enumerate(in_landmarks):
        # if ind in [1,2,3,4,5,6,9,10]:
        #     continue
        model_input.append(joint.x)
        model_input.append(joint.y)
        # model_input.append(joint.visibility)
    return model_input


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


def get_min_visbility(scelton_points: np.ndarray):
    visability_vector = []
    for side in [True, False]:
        visability_vector.extend(get_min_visability(scelton_points, side))
    return visability_vector


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
