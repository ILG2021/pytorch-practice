import click
import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, fl_stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=fl_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()
        # shortcut是保证x和out的维度（c h w）一致，能够相加
        if fl_stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, stride=fl_stride),
                                          nn.BatchNorm2d(out_channel))  # kernel 3 padding 1和kernel 1 padding 0计算后宽高一致

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.stage1 = nn.Sequential(ResBlock(16, 16, 1),
                                    ResBlock(16, 16, 1),
                                    ResBlock(16, 16, 1))
        self.stage2 = nn.Sequential(ResBlock(16, 32, 2),
                                    ResBlock(32, 32, 1),
                                    ResBlock(32, 32, 1))
        self.stage3 = nn.Sequential(ResBlock(32, 64, 2),
                                    ResBlock(64, 64, 1),
                                    ResBlock(64, 64, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet20().to(device)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5),
]
)

train_set = torchvision.datasets.CIFAR10("./data", True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10("./data", False, transform=transform, download=True)

train_loader = DataLoader(train_set, 128, True)
test_loader = DataLoader(test_set, 128, False)


def train():
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, [80, 120], 0.1)
    model.train()
    for epoch in range(200):
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            predict = model(image)
            loss = loss_fn(predict, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f"epoch {epoch + 1} loss: {loss.item()}")
    torch.save(model.state_dict(), "./basic/resnet20.pth")


def infer():
    model.load_state_dict(torch.load("./basic/resnet20.pth"))
    model.eval()
    hit = 0
    total = 0
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        predict = model(image)
        hit += (torch.argmax(predict, -1) == label).sum().item()
        total += image.shape[0]
    print(f"Accurate: {hit / total}")


@click.command()
@click.option("--is_train", default=False)
def main(is_train):
    if is_train:
        train()
    else:
        infer()


if __name__ == '__main__':
    main()
