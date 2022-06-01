import torch
from torch import nn, optim
from tools.process_csv import CsvDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from genric_net.genric_net import Network

error_csv_path = r"C:\git_repos\project_noam_nofar\csv_files\new\down.csv"
pose_datasets = CsvDataset(file=error_csv_path)
# pose_datasets.make_classes_samples_eq()
pose_datasets.add_miror()
# pose_datasets.error_process()
pose_datasets.vec_dir_process()
train_dataset, validation_dataset = pose_datasets.df_to_datasets('class')
model = Network(32, 2, [48, 33, 12], drop_p=0.5)


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
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 4

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
# torch.save(model.state_dict(), 'checkpoint_angle20.4.pth')
checkpoint = {'input_size': 32,
              'output_size': 2,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'down_test_new.pth')
plt.show()

