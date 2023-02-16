import torch
from torch.utils.data import Dataset
import numpy as np

# Load Sample and Label
class train_Loader(Dataset):

    def __init__(self, train_npy_file, label_npy_file):
        self.all_curve = (np.load(train_npy_file)).astype(float)
        self.all_label = (np.load(label_npy_file)).astype(float)

    def __getitem__(self, index):
        curve = torch.tensor(self.all_curve[index])
        label = torch.tensor(self.all_label[index])
        return curve, label

    def __len__(self):
        return len(self.all_curve)

# Without Label
class pred_Loader(Dataset):

    def __init__(self, pred_npy_file):
        self.all_curve = (np.load(pred_npy_file)).astype(float)

    def __getitem__(self, index):
        curve = torch.tensor(self.all_curve[index])
        return curve

    def __len__(self):
        return len(self.all_curve)

if __name__ == "__main__":

    # load dataset, 'xxx.npy' has been flatten with the shape of [pixels, wavelength]
    train_dataset = train_Loader(r'xx/train_sample.npy', r'xx/train_label.npy')
    print("number of loaded train curves：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
    for curve, label in train_loader:
        # get the length of this batch, especially for the last batch, may less than 'batch_size'
        len_of_this_batch = curve.shape[0]
        # before input the Net, it should add one dim(channel) after dim(batch)
        print(curve.reshape(len_of_this_batch, 1, -1).shape)
        break

