from model import dataset
from .part import *


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = head_block()
        self.backbone = backbone_block()
        self.dense = dense_block()

    def forward(self, x):
        x = self.head(x)
        x = self.backbone(x)
        x = self.dense(x)
        return x


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet().to(device=device)

    from torchsummary import summary
    summary(net, input_size=(1, 120)) # (channel, H, W). Here, (1 channel, 120 bands)

    train_dataset = dataset.train_Loader(r'xx/train_sample.npy', r'xx/train_label.npy')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
    for curve, label in train_loader:
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        print(curve.shape)
        out = net(curve)
        print(out.shape)
        break