import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
class CreditDataSet(Dataset):

    def __init__(self, path):
        full_data = pd.read_csv(path)
        full_data['Amount'] = StandardScaler().fit_transform(full_data['Amount'].values.reshape(-1, 1))
        full_features_train = full_data[full_data['Class'] == 0]
        full_features_test = full_data[full_data['Class'] == 1]
        print(full_features_test.shape)
        full_features_train = full_features_train.drop(["Time", "Class"], axis=1)
        full_features_test = full_features_test.drop(["Time", "Class"], axis=1)

        train, val = train_test_split(full_features_train, test_size=0.2, random_state=1234)

        self.df = {
                 'train': train,
                 'val' : val,
                 'test': full_features_test
        }
        self._target = self.df['train']

    def set_split(self, split):
        self._target = self.df[split]

    def __len__(self):
        return self._target.shape[0]

    def __getitem__(self, index):
        row = self._target.iloc[index]
        feature_array = np.fromiter(row.to_dict().values(), float)
        return {
            'X': torch.from_numpy(feature_array).float(),
            'Y': torch.from_numpy(feature_array).float()
        }

class FraudDetectionModel(nn.Module):
    def __init__(self):
        super(FraudDetectionModel, self).__init__()
        self.encoder = nn.Sequential(
             nn.Linear(29, 14),
             nn.ReLU(True),
             nn.Linear(14, 7),
             nn.ReLU(True),
            nn.Linear(7,3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,7),
            nn.ReLU(True),
            nn.Linear(7, 14),
            nn.ReLU(True),
            nn.Linear(14, 29),
            nn.Tanh()
        )
    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        return x
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def compute_accuracy(y_pred, y_target):
    y_pred_indices = y_pred.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


num_epochs = 500
batch_size = 128
learning_rate = 1e-3
csv_path = "~/PycharmProjects/atap/resources/creditcard.csv"
dataset = CreditDataSet(csv_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = FraudDetectionModel().cpu()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)
dataset.set_split('train')
for epoch in range(num_epochs):
    running_loss = 0.
    batch_generator = generate_batches(dataset,
                                       batch_size=batch_size)
    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        # ===================forward=====================
        output = model(torch.randn(batch_dict['X'].shape) + batch_dict['X'])
        loss = criterion(output, batch_dict['Y'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
        # ===================backward====================
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, running_loss))
torch.save(model, 'autoencoder_with_noise')
# model = torch.load('autoencoder')
# dataset.set_split('test')
# batch_generator = generate_batches(dataset,
#                                    batch_size=batch_size,
#                                    device='cpu')
# running_loss = 0.
# running_acc = 0.
# model.eval()
# running_loss = 0.
# for batch_index, batch_dict in enumerate(batch_generator):
#     print(batch_dict['X'][0])
#     # compute the output
#     y_pred = model(batch_dict['X'][0])
#     print("Y_PRED", y_pred)
#     # step 3. compute the loss
#     loss = criterion(y_pred, batch_dict['X'][0])
#     print(loss)
#     loss_t = loss.item()
#     running_loss += (loss_t - running_loss) / (batch_index + 1)
#     print('loss:{:.4f}'
#           .format(running_loss))