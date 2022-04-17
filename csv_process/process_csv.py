import pandas as pd
import pickle  # Object serialization.
import tensorflow as tf
from tools.get_feature_vec import *


class CsvDataset:

    def __init__(self, file):
        # self.dataframe = pd.read_csv(file)
        self.dataframe = pd.read_csv(file)
        self.val_df = None
        self.train_df = None
        self.val_ds = None
        self.train_ds = None

    # TODO use less points [0,11-32]
    def csv_preprocessing(self):
        ''''Remove points 1 to points 11 of columns.'''
        # headers = [*pd.read_csv(dataset_csv_file, nrows=1)]
        # df1 = pd.read_csv(dataset_csv_file, usecols=[c for c in headers if c != 'name'])
        df2 = self.dataframe.copy()
        columns_removed = [
            'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
            'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
            'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11',
            'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11',

            # 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21',
            # 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31',
            # 'v32', 'v33',
        ]
        for ind, joint in enumerate(left_joint_list):
            a = np.array([df2['x' + str(joint[0])], df2['y' + str(joint[0])]])
            b = np.array([df2['x' + str(joint[1])], df2['y' + str(joint[1])]])
            c = np.array([df2['x' + str(joint[2])], df2['y' + str(joint[2])]])
            df2['angleL' + str(ind)] = cucl_angle(a,b,c)
            df2['vis_angleL' + str(ind)] = np.min([df2['v' + str(joint[0])], df2['v' + str(joint[1])], df2['v' + str(joint[2])]])
        for ind, joint in enumerate(right_joint_list):
            a = np.array([df2['x' + str(joint[0])], df2['y' + str(joint[0])]])
            b = np.array([df2['x' + str(joint[1])], df2['y' + str(joint[1])]])
            c = np.array([df2['x' + str(joint[2])], df2['y' + str(joint[2])]])
            df2['angleR' + str(ind)] = cucl_angle(a,b,c)
            df2['vis_angleR' + str(ind)] = np.min([df2['v' + str(joint[0])], df2['v' + str(joint[1])], df2['v' + str(joint[2])]])
        df2 = df2.drop(columns_removed, axis='columns')

        # print('*'*60)
        # print(f'df1: \n{self.dataframe.head()}')
        # print(f'df2: \n{df2.head()}')
        return df2

    def df_to_datasets(self, dataframe, target):
        ''''Input pandas dataframe to get specific features dataset of datasets.'''

        # frac(float): 要抽出的比例, random_state：隨機的狀態.
        self.val_df = dataframe.sample(frac=0.2, random_state=1337)
        # drop the colum 1 of 'class'.
        self.train_df = dataframe.drop(self.val_df.index)

        train_df = self.train_df.copy()
        val_df = self.val_df.copy()

        train_labels = train_df.pop(target)
        val_labels = val_df.pop(target)

        # tf.data.Dataset.from_tensor_slices(): 可以獲取列表或數組的切片。
        self.train_ds = tf.data.Dataset.from_tensor_slices((dict(train_df), train_labels))
        self.val_ds = tf.data.Dataset.from_tensor_slices((dict(val_df), val_labels))

        # shuffle(): 用來打亂數據集中數據順序.
        # buffer_size: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/661458/
        self.train_ds = self.train_ds.shuffle(buffer_size=len(self.train_ds))
        self.val_ds = self.val_ds.shuffle(buffer_size=len(self.val_ds))

        return self.train_ds, self.val_ds


def load_dataset(csv_data):
    df = pd.read_csv(csv_data)
    # print(f'Top5 datas: \n{df.head()}')
    # print(f'Last5 datas: \n{df.tail()}')
    # print(f'Specific class: \n', df[df['class']=='bridge'])  # Show specific class data.

    features = df.drop('class', axis=1)  # Features, drop the colum 1 of 'class'.
    target_value = df['class']  # target value.

    x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.3, random_state=1234)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    data_path = r"C:\git_repos\project_noam_nofar\count_with_engle_trash_hold\coords.csv"
    pose_datasets = CsvDataset(file=data_path)
    df_pose = pose_datasets.csv_preprocessing()
