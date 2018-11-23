from lightai.core import *


class Model(nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.base = self.get_base()
        self.dummy_forward(sz)
        self.mean = T(np.array([20.50361 , 13.947072, 13.408824, 21.106398]).reshape((-1, 1, 1))).half()
        self.std = T(np.array([38.12811 , 39.742226, 28.598948, 38.173912]).reshape((-1, 1, 1))).half()
    def forward(self, x):
        x = x.half()
        x = (x-self.mean)/self.std
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc_bn(torch.relu(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    def dummy_forward(self, sz):
        with torch.no_grad():
            self.base.eval()
            x = torch.zeros(1, 4, sz, sz)
            x = self.base(x)
            width = x.shape[-1]
            self.avgpool = nn.AvgPool2d(2)
            self.fc1 = nn.Linear(512*(width//2)**2, 128)
            self.fc_bn = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 28)
    def get_base(self):
        resnet = torchvision.models.resnet18(pretrained=True)
        conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        conv1.weight.data[:, :-1] = resnet.conv1.weight.data
        conv1.weight.data[:, -1] = resnet.conv1.weight.data.mean(dim=1)
        resnet.conv1 = conv1
        return nn.Sequential(*list(resnet.children())[:-2])