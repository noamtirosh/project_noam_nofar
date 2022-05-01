import torch
import pandas as pd

from genric_net.genric_net import load_network
import numpy as np
from tools.get_feature_vec import get_angles, get_min_visability, get_angle_vec_input
from tools.process_csv import CsvDataset

# init model
# model_path = r"C:\git_repos\project_noam_nofar\train_model\checkpoint_angle20.4.pth"
# checkpoint = torch.load(model_path)
# # model.load_state_dict(checkpoint['state_dict'])
# model.load_state_dict(checkpoint)

model_path = r"C:\git_repos\project_noam_nofar\models\work very good\model_angle_vec_20.4.pth"
model = load_network(model_path)
data_csv_file = r"C:\git_repos\project_noam_nofar\csv_files\old\up_down_classifiction.csv"
roc_df = pd.DataFrame()
pose_datasets = CsvDataset(file=data_csv_file)
pose_datasets.angle_and_vec_process()
roc_df = pose_datasets.data.copy()
roc_df.pop('class')
roc_df = torch.tensor(roc_df.values, dtype=torch.float)
model.eval()
with torch.no_grad():
    out = model(roc_df)
ps = torch.exp(out)
ps = ps.cpu().detach().numpy()
ps_df = pd.DataFrame(ps, columns=['class_1', 'class_2'])
roc_df = pose_datasets.data.copy()
roc_df = roc_df.append(ps_df)
roc_df.to_csv("roc_out.csv")

