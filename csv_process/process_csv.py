import pandas as pd
from tools.get_feature_vec import *
import torch
from torch.utils.data import TensorDataset, DataLoader


class CsvDataset:

    def __init__(self, file):
        # self.dataframe = pd.read_csv(file)
        self.dataframe = pd.read_csv(file)
        self.data = None
        self.val_df = None
        self.train_df = None
        self.val_ds = None
        self.train_ds = None

    # TODO use less points [0,11-32]
    def angle_process(self):
        df = self.dataframe
        angel_data = pd.DataFrame()
        angel_data['class'] = df['class']
        for ind, joint in enumerate(left_joint_list):
            a = np.array([df['x' + str(joint[0])], df['y' + str(joint[0])]])
            b = np.array([df['x' + str(joint[1])], df['y' + str(joint[1])]])
            c = np.array([df['x' + str(joint[2])], df['y' + str(joint[2])]])
            angel_data['angleL' + str(ind)] = culc_angle(a, b, c) / 360  # for normalize
            angel_data['vis_angleL' + str(ind)] = np.min(
                [df['v' + str(joint[0])], df['v' + str(joint[1])], df['v' + str(joint[2])]])
        for ind, joint in enumerate(right_joint_list):
            a = np.array([df['x' + str(joint[0])], df['y' + str(joint[0])]])
            b = np.array([df['x' + str(joint[1])], df['y' + str(joint[1])]])
            c = np.array([df['x' + str(joint[2])], df['y' + str(joint[2])]])
            angel_data['angleR' + str(ind)] = culc_angle(a, b, c) / 360  # for normalize
            angel_data['vis_angleR' + str(ind)] = np.min(
                [df['v' + str(joint[0])], df['v' + str(joint[1])], df['v' + str(joint[2])]])
        self.data = angel_data.copy()

    def point_loc_process(self):
        ''''Remove points 1 to points 11 of columns.'''
        # headers = [*pd.read_csv(dataset_csv_file, nrows=1)]
        # df1 = pd.read_csv(dataset_csv_file, usecols=[c for c in headers if c != 'name'])
        xyz_data = self.dataframe.copy()

        columns_removed = [
            'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
            'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
            'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11',
            'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11',

            # 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21',
            # 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31',
            # 'v32', 'v33',
        ]
        xyz_data.drop(columns_removed, axis='columns', inplace=True)
        xyz_data.drop(xyz_data.filter(regex='z').columns, axis='columns', inplace=True)
        self.data = xyz_data.copy()

    def df_to_datasets(self, target):
        ''''Input pandas dataframe to get specific features dataset of datasets.'''

        # frac(float): 要抽出的比例, random_state：隨機的狀態.
        self.val_df = self.data.sample(frac=0.2, random_state=1337)
        self.train_df = self.data.drop(self.val_df.index)
        train_df = self.train_df.copy()
        val_df = self.val_df.copy()

        # drop the colum 1 of 'class'.
        train_labels = train_df.pop(target)
        val_labels = val_df.pop(target)
        tensor_train = torch.tensor(train_df.values)
        tensor_val = torch.tensor(val_df.values)

        tensor_train_labels = torch.tensor(train_labels.values)
        tensor_val_labels = torch.tensor(val_labels.values)
        self.train_ds = TensorDataset(tensor_train, tensor_train_labels)
        self.val_ds = TensorDataset(tensor_val, tensor_val_labels)
        return DataLoader(self.train_ds, batch_size=32, shuffle=True), DataLoader(self.val_ds, batch_size=32, shuffle=True)


def load_dataset(csv_data):
    df = pd.read_csv(csv_data)
    # print(f'Top5 datas: \n{df.head()}')
    # print(f'Last5 datas: \n{df.tail()}')
    # print(f'Specific class: \n', df[df['class']=='bridge'])  # Show specific class data.

    features = df.drop('class', axis=1)  # Features, drop the colum 1 of 'class'.
    target_value = df['class']  # target value.


if __name__ == '__main__':
    data_path = r"C:\Users\noam\Videos\project\currect\data1.csv"
    pose_datasets = CsvDataset(file=data_path)
    pose_datasets.angle_process()
    train_data_loader,val_data_loader = pose_datasets.df_to_datasets('class')
