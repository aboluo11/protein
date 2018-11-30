from lightai.core import *


class Model(nn.Module):
    def __init__(self, base=torchvision.models.resnet18, pretrained=True):
        super().__init__()
        self.base = self.get_base(base, pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 28)
        self.mean = T(np.array(
            [22.4812, 13.4036, 14.9883]).reshape((-1, 1, 1))).half()
        self.std = T(np.array([38.5135, 27.0687, 40.9111]).reshape(
            (-1, 1, 1))).half()

    def forward(self, x):
        x = x.half()
        x = x.permute(0, 3, 1, 2)
        x = (x-self.mean)/self.std
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def get_base(self, base, pretrained):
        resnet = base(pretrained=pretrained)
        # conv1 = nn.Conv2d(4, 64, kernel_size=(
        #     7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # conv1.weight.data[:, :-1] = resnet.conv1.weight.data
        # conv1.weight.data[:, -1] = resnet.conv1.weight.data.mean(dim=1)
        # resnet.conv1 = conv1
        return nn.Sequential(*list(resnet.children())[:-2])
