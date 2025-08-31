import click
import torch.utils.data
import torchvision.datasets
from torch import nn
from torch.functional import F
from torchvision import transforms


class LeNet5(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 6, 5, padding='same')
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = F.avg_pool2d(x, 2, stride=2)
        x = F.sigmoid(self.conv2(x))
        x = F.avg_pool2d(x, 2, stride=2)
        x = x.view(x.shape[0], -1)  # shape[0]是batch size
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST("./data", True, transform=transform, download=True)
testset = torchvision.datasets.MNIST("./data", False, transform=transform, download=True)

trainloader = torch.utils.data.DataLoader(trainset, 64, True)
testloader = torch.utils.data.DataLoader(testset, 64, False)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    model = LeNet5().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(50):
        for image, label in trainloader:
            image, label = image.to(device), label.to(device)
            predict = model(image)
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"epoch: {epoch} loss: {loss.item()}")
    torch.save(model.state_dict(), "basic/lenet5.pth")


def infer():
    model = LeNet5().to(device)
    model.eval()
    model.load_state_dict(torch.load("./basic/lenet5.pth"))
    hit = 0
    total = 0
    for image, label in testloader:
        image, label = image.to(device), label.to(device)
        predict = model(image)
        predict = torch.argmax(predict, -1)
        hit += (predict == label).sum().item()
        total += label.size(0)
    print(f"accurate:{hit / total}")


@click.command()
@click.option("--is_train", default=False)
def main(is_train):
    if is_train:
        train()
    else:
        infer()


if __name__ == '__main__':
    main()
