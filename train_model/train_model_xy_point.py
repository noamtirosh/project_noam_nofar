import torch
from torch import nn, optim
from tools.process_csv import CsvDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(66, 22)
        self.fc2 = nn.Linear(22, 12)
        self.fc3 = nn.Linear(12, 2)
        # self.fc4 = nn.Linear(64, 10)
        # Dropout module with 0.3 drop probability
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x



if __name__ == '__main__':
    data_csv_file = r"C:\Users\noam\Videos\project\currect\data1.csv"
    pose_datasets = CsvDataset(file=data_csv_file)
    #pose_datasets.add_miror()
    pose_datasets.point_loc_process()
    train_dataset, validation_dataset = pose_datasets.df_to_datasets('class')
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_data_loader =  DataLoader(validation_dataset, batch_size=64, shuffle=True)

    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 70

    train_losses, test_losses = [], []
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
    torch.save(model.state_dict(), 'checkpoint_xyz.pth')


