import torch
from torch import nn, optim
from tools.process_csv import CsvDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

RIGHT_SIDE = 0
LEFT_SIDE = 1
FRONT_SIDE = 2


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*17, 16*17)
        self.fc2 = nn.Linear(16*17, 8*17)
        self.fc3 = nn.Linear(8*17, 4*17)
        self.fc4 = nn.Linear(4*17, 3)
        # Dropout module with 0.3 drop probability
        self.dropout = nn.Dropout(p=0.3)
        self.batch = nn.BatchNorm1d(32)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # make sure input tensor is flattened
        matrix_in = np.zeros([x.shape[0], 2, 17])
        matrix_in[:, 0, :] = x[:, 0:34:2]
        matrix_in[:, 1, :] = x[:, 1:34:2]

        x = torch.Tensor(matrix_in)
        # x = x.view(x.shape[0], -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(self.batch(x))

        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        # x = self.fc1(x)
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))


        x = self.fc4(x)

        x = F.log_softmax(x, dim=1)

        return x


if __name__ == '__main__':

    error_csv_path = r"C:\git_repos\project_noam_nofar\csv_files\one_side_csv\correct_right.csv"
    error_csv_path = r"C:\git_repos\project_noam_nofar\csv_files\one_side_csv\count_with_err_right.csv"

    pose_datasets = CsvDataset(file=error_csv_path)
    # pose_datasets.make_classes_samples_eq()
    pose_datasets.point_loc_process_by_side(RIGHT_SIDE)
    train_dataset, validation_dataset = pose_datasets.df_to_datasets('class')




    # model = Network(32, 3, [48, 33, 12], drop_p=0.5)
    model = Classifier()
    # data_csv_file = r"C:\git_repos\project_noam_nofar\csv_files\up_down_classifiction.csv"
    # pose_datasets = CsvDataset(file=data_csv_file)
    # pose_datasets.add_miror()
    # pose_datasets.angle_and_vec_process()
    # num_output = pose_datasets.get_num_class()
    # train_dataset, validation_dataset = pose_datasets.df_to_datasets('class')
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    criterion = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 8

    train_losses, test_losses, accuracy = [], [], []
    for e in range(epochs):
        tot_train_loss = 0
        for images, labels in train_data_loader:
            optimizer.zero_grad()

            log_ps = model(images)

            loss = criterion(log_ps, labels)
            tot_train_loss += loss.item()

            loss.backward()
            optimizer.step()
        else:
            tot_test_loss = 0
            test_correct = 0  # Number of correct predictions on the test set

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in validation_data_loader:
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    tot_test_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_correct += equals.sum().item()

            # Get mean loss to enable comparison between train and test sets
            train_loss = tot_train_loss / len(train_data_loader.dataset)
            test_loss = tot_test_loss / len(validation_data_loader.dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracy.append(test_correct / len(validation_data_loader.dataset))
            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Test Loss: {:.3f}.. ".format(test_loss),
                  "Test Accuracy: {:.3f}".format(test_correct / len(validation_data_loader.dataset)))

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    # checkpoint = {'input_size': 24,
    #               'output_size': 2,
    #               'hidden_layers': 12,
    #               'state_dict': model.state_dict()}
    # torch.save(checkpoint, 'checkpoint.pth')
    torch.save(model.state_dict(), 'conv_count_with_err.pth')
    # checkpoint = {'input_size': 32,
    #               'output_size': 3,
    #               'hidden_layers': [each.out_features for each in model.hidden_layers],
    #               'state_dict': model.state_dict()}
    #
    # torch.save(checkpoint, 'down_model.pth')
    plt.show()
    plt.plot(accuracy, label='accuracy')
    plt.legend(frameon=False)
    plt.title('Accuracy')

    plt.show()
