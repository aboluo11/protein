from lightai.core import *


mean = T(np.array(
    [9.1536, 9.9506, 1.9361, 24.4079]).reshape((-1, 1, 1))).half()
std = T(np.array([13.1594, 17.5587, 5.0485, 36.4777]).reshape(
    (-1, 1, 1))).half()


class Model(nn.Module):
    def __init__(self, base=torchvision.models.resnet18, pretrained=True):
        super().__init__()
        self.base = self.get_base(base, pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 28)

    def forward(self, x):
        x = x.half()
        x = x.permute(0, 3, 1, 2)
        x = (x-mean)/std
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def get_base(self, base, pretrained):
        resnet = base(pretrained=pretrained)
        conv1 = nn.Conv2d(4, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        conv1.weight.data[:, :-1] = resnet.conv1.weight.data
        conv1.weight.data[:, -1] = resnet.conv1.weight.data.mean(dim=1)
        resnet.conv1 = conv1
        return nn.Sequential(*list(resnet.children())[:-2])


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(4, 64, kernel_size=7, padding=3, stride=2)
        self.conv2 = ConvBlock(64, 128, stride=2)
        self.conv3 = ConvBlock(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 28)

    def forward(self, x):
        x = x.half()
        x = x.permute(0, 3, 1, 2)
        x = (x-mean)/std
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        return x
