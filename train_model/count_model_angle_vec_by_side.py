import torch
from torch import nn, optim
from tools.process_csv import CsvDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from genric_net.genric_net import Network

# use only csv of the right side for this train
# in this model we assume that the user is with his side to the camera and we get only half of the point
RIGHT_SIDE = 0
LEFT_SIDE = 1

model = Network(14, 2, [48, 24, 12], drop_p=0.3)

if __name__ == '__main__':
    right_side_data_csv_file = r"C:\git_repos\project_noam_nofar\csv_files\one_side_csv\count_model_csv\correct_left.csv"
    pose_datasets = CsvDataset(file=right_side_data_csv_file)
    # pose_datasets.make_classes_samples_eq()
    pose_datasets.count_model_one_side_process(LEFT_SIDE)
    train_dataset, validation_dataset = pose_datasets.df_to_datasets('class')
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 10

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
            tp = 0
            tn = 0
            fn = 0
            fp = 0
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in validation_data_loader:
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    tot_test_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    train_labels = labels.view(*top_class.shape)
                    tp += (top_class[train_labels == 1] == 1).sum().item()
                    tn += (top_class[train_labels == 0] == 0).sum().item()
                    fp += (top_class[train_labels == 1] == 0).sum().item()
                    fn += (top_class[train_labels == 0] == 1).sum().item()

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
                  "Test Accuracy: {:.3f}".format(test_correct / len(validation_data_loader.dataset)),
                  "TP: {} , TN: {} ,FN: {} ,FP: {}".format(tp, tn, fn, fp))

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.title('loss')
    plt.savefig("lass_graph Test_Loss_{:.3f}_TP_{}_TN_{}_FN_{}_FP_{}.png".format(test_loss,tp, tn, fn, fp),format='png')
    plt.show()
    plt.figure()
    plt.plot(accuracy, label='accuracy')
    plt.legend(frameon=False)
    plt.title('Accuracy')

    plt.show()
    # checkpoint = {'input_size': 24,
    #               'output_size': 2,
    #               'hidden_layers': 12,
    #               'state_dict': model.state_dict()}
    # torch.save(checkpoint, 'checkpoint.pth')
    # torch.save(model.state_dict(), 'checkpoint_angle20.4.pth')
    checkpoint = {'input_size': 14,
                  'output_size': 2,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'count_model_left_side.pth')
