import pandas as pd
from tools.get_feature_vec import *
import torch
from torch.utils.data import TensorDataset
from get_mediapipe_example_vec import FullBodyPoseEmbedder

NUM_OF_LANDMARKS = 33


class CsvDataset:

    def __init__(self, file):
        # self.dataframe = pd.read_csv(file)
        self.dataframe = pd.read_csv(file)
        self.data = None
        self.val_df = None
        self.train_df = None
        self.val_ds = None
        self.train_ds = None

    def add_miror(self):
        miror_df = self.dataframe.copy()
        for i in range(7, 32, 2):
            miror_df['x' + str(i)] = self.dataframe['x' + str(i + 1)]
            miror_df['y' + str(i)] = self.dataframe['y' + str(i + 1)]
            miror_df['z' + str(i)] = self.dataframe['z' + str(i + 1)]
            miror_df['v' + str(i)] = self.dataframe['v' + str(i + 1)]
        for i in range(8, 33, 2):
            miror_df['x' + str(i)] = self.dataframe['x' + str(i - 1)]
            miror_df['y' + str(i)] = self.dataframe['y' + str(i - 1)]
            miror_df['z' + str(i)] = self.dataframe['z' + str(i - 1)]
            miror_df['v' + str(i)] = self.dataframe['v' + str(i - 1)]
        self.dataframe = self.dataframe.append(miror_df)

    # TODO use less points [0,11-32]
    def angle_process(self):
        df = self.dataframe
        angel_data = pd.DataFrame()
        angel_data['class'] = df['class']
        self.__add_angle_to_df(angel_data, df)
        self.data = angel_data.copy()

    def angle_and_vec_process(self):
        df = self.dataframe
        combine_data = pd.DataFrame()
        combine_data['class'] = df['class']
        self.__add_angle_to_df(combine_data, df)
        self.__add_vec_dir_to_df(combine_data, df)
        self.data = combine_data.copy()

    def __add_angle_to_df(self, add_to, data):
        for ind, joint in enumerate(left_joint_list):
            a = np.array([data['x' + str(joint[0])], data['y' + str(joint[0])]])
            b = np.array([data['x' + str(joint[1])], data['y' + str(joint[1])]])
            c = np.array([data['x' + str(joint[2])], data['y' + str(joint[2])]])
            add_to['angleL' + str(ind)] = culc_angle(a, b, c) / 360  # for normalize
            add_to['vis_angleL' + str(ind)] = np.min(
                [data['v' + str(joint[0])], data['v' + str(joint[1])], data['v' + str(joint[2])]])
        for ind, joint in enumerate(right_joint_list):
            a = np.array([data['x' + str(joint[0])], data['y' + str(joint[0])]])
            b = np.array([data['x' + str(joint[1])], data['y' + str(joint[1])]])
            c = np.array([data['x' + str(joint[2])], data['y' + str(joint[2])]])
            add_to['angleR' + str(ind)] = culc_angle(a, b, c) / 360  # for normalize
            add_to['vis_angleR' + str(ind)] = np.min(
                [data['v' + str(joint[0])], data['v' + str(joint[1])], data['v' + str(joint[2])]])

    def point_loc_process(self):
        ''''Remove points 1 to points 11 of columns.'''
        # headers = [*pd.read_csv(dataset_csv_file, nrows=1)]
        # df1 = pd.read_csv(dataset_csv_file, usecols=[c for c in headers if c != 'name'])
        xyz_data = self.dataframe.copy()

        columns_removed = [
            'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
            'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
            'z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10',
            'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',

            # 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21',
            # 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31',
            # 'v32', 'v33',
        ]
        # xyz_data.drop(columns_removed, axis='columns', inplace=True)
        xyz_data.drop(xyz_data.filter(regex='z').columns, axis='columns', inplace=True)
        xyz_data.drop(xyz_data.filter(regex='v').columns, axis='columns', inplace=True)
        self.data = xyz_data.copy()

    def vec_dir_process(self):
        df = self.dataframe
        vec_data = pd.DataFrame()
        vec_data['class'] = df['class']
        self.__add_vec_dir_to_df(vec_data, df)
        self.data = vec_data.copy()

    def media_pipe_example_process(self):
        landmarks = np.zeros([NUM_OF_LANDMARKS, 3, self.dataframe.shape[0]])
        for i in range(0, NUM_OF_LANDMARKS):
            landmarks[i, 0, :] = self.dataframe['x' + str(i)]
            landmarks[i, 1, :] = self.dataframe['y' + str(i)]
            landmarks[i, 2, :] = self.dataframe['z' + str(i)]
        body_pose = FullBodyPoseEmbedder()
        body_pose_output = body_pose(landmarks)
        body_pose_output = np.swapaxes(body_pose_output, 2, 0)
        mp_data = pd.DataFrame()
        mp_data['class'] = self.dataframe['class']
        for cord in range(3):
            mp_data = pd.concat([mp_data, pd.DataFrame(body_pose_output[:, cord, :])], axis=1)
        self.data = mp_data.copy()

    def error_process(self):
        df = self.dataframe
        data = pd.DataFrame()
        data['class'] = df['class']
        self.__add_vec_dir_to_df(data, df)
        self.__add_cross_vec_dir_to_df(data, df)
        self.data = data.copy()

    @staticmethod
    def __add_vec_dir_to_df(add_to, data):
        # left side
        for ind, two_joints in enumerate(left_tow_joints_list):
            a = np.array([data['x' + str(two_joints[0])], data['y' + str(two_joints[0])]])
            b = np.array([data['x' + str(two_joints[1])], data['y' + str(two_joints[1])]])
            add_to['vec_dir' + str(ind)] = culc_vec_direction(a, b) / 360
            add_to['vis_vec_dir' + str(ind)] = np.min(
                [data['v' + str(two_joints[0])], data['v' + str(two_joints[1])]])
        # right side
        for ind, two_joints in enumerate(right_tow_joints_list):
            a = np.array([data['x' + str(two_joints[0])], data['y' + str(two_joints[0])]])
            b = np.array([data['x' + str(two_joints[1])], data['y' + str(two_joints[1])]])

            add_to['vec_dir' + str(ind + len(left_tow_joints_list))] = culc_vec_direction(a, b) / 360
            add_to['vis_vec_dir' + str(ind + len(left_tow_joints_list))] = np.min(
                [data['v' + str(two_joints[0])], data['v' + str(two_joints[1])]])

    @staticmethod
    def __add_cross_vec_dir_to_df(add_to, data):
        for ind, two_joint_cross in enumerate(cross_body_joints_list):
            a = np.array([data['x' + str(two_joint_cross[0])], data['y' + str(two_joint_cross[0])]])
            b = np.array([data['x' + str(two_joint_cross[1])], data['y' + str(two_joint_cross[1])]])
            add_to['cross_vec_dir' + str(ind)] = culc_vec_direction(a, b) / 360
            add_to['vis_cross_vec_dir' + str(ind)] = np.min(
                [data['v' + str(two_joint_cross[0])], data['v' + str(two_joint_cross[1])]])

    def combine_process(self):
        self.point_loc_process()
        self.__add_angle_to_df(self.data, self.data)

    def df_to_datasets(self, target, val_prc=0.2):
        ''''Input pandas dataframe to get specific features dataset of datasets.'''

        # frac(float): 要抽出的比例, random_state：隨機的狀態.
        self.val_df = self.data.sample(frac=val_prc, random_state=1337)
        self.train_df = self.data.drop(self.val_df.index)
        train_df = self.train_df.copy()
        val_df = self.val_df.copy()

        # drop the colum 1 of 'class'.
        train_labels = train_df.pop(target)
        val_labels = val_df.pop(target)
        tensor_train = torch.tensor(train_df.values, dtype=torch.float)
        tensor_val = torch.tensor(val_df.values, dtype=torch.float)

        tensor_train_labels = torch.tensor(train_labels.values, dtype=torch.long)
        tensor_val_labels = torch.tensor(val_labels.values, dtype=torch.long)
        return TensorDataset(tensor_train, tensor_train_labels), TensorDataset(tensor_val, tensor_val_labels)

    def get_num_class(self):
        return self.dataframe["class"].nunique()

    def make_classes_samples_eq(self):
        df = self.dataframe
        new_df = pd.DataFrame()
        num_of_class = int(df["class"].nunique())
        min_num_samples = df[df["class"] == 0]["class"].count()
        for i in range(1, num_of_class):
            min_num_samples = min(df[df["class"] == i]["class"].count(), min_num_samples)
        for i in range(num_of_class):
            new_df = new_df.append(df[df["class"] == i].sample(n=min_num_samples))
        self.dataframe = new_df


if __name__ == '__main__':
    data_path = r"C:\Users\noam\Videos\project\currect\data1.csv"
    pose_datasets = CsvDataset(file=data_path)
    pose_datasets.angle_process()
    train_data_loader, val_data_loader = pose_datasets.df_to_datasets('class')
