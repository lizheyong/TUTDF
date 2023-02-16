from model.model import *
from model.dataset import train_Loader
from torch import optim
import torch


def train_net(net, train_dataset, device, batch_size, lr, epochs):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, eps=1e-08,  weight_decay=1e-2,  momentum=0.9)
    best_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        for curve, label in train_loader:
            optimizer.zero_grad()
            curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            out = net(curve)
            loss = criterion(out, label)
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), fr"xxx.pth")
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch}/{epochs}, loss:{loss.item()}')
    print(f'best_loss:{best_loss.item()}')


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ResNet = ResNet().to(device=device)
    dataset = train_Loader(fr"xxx\train.npy", fr"xxx\train_label.npy")
    train_net(net=ResNet, train_dataset=dataset, device=device,
              batch_size=1024, lr=0.0001, epochs=300)
