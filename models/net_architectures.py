import torch
import torch.nn as nn

class VGGInverterG(nn.Module):
    def __init__(self, nc=3):
        super(VGGInverterG, self).__init__()
        self.conv = nn.Sequential(
            # 14 -> 28
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 28 -> 56
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 56 -> 112
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 112 -> 224
            nn.ConvTranspose2d(64, nc, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        # input: (N, 100)
        out = self.conv(input)
        return out


class VGGInverterD(nn.Module):
    def __init__(self, nc=3, input_size=784):
        super(VGGInverterD, self).__init__()
        self.conv = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(nc, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # 112 -> 56
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 56 -> 28
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # 28 -> 14
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.dense = nn.Sequential(
            # 14 -> 1
            nn.Linear(512 * 14 * 14, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # input: (N, nc, 224, 224)
        out = self.conv(input)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return out


class VGGInverterDSpectral(nn.Module):
    def __init__(self, nc=3, input_size=784):
        super(VGGInverterDSpectral, self).__init__()

        # 224 -> 112
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nc, 64, 3, stride=2, padding=1))
        # 112 -> 56
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1))
        # 56 -> 28
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1))
        # 28 -> 14
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, stride=2, padding=1))

        self.fc = nn.utils.spectral_norm(nn.Linear(512 * 14 * 14, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input: (N, nc, 224, 224)
        out = input
        out = nn.LeakyReLU(0.2)(self.conv1(out))
        out = nn.LeakyReLU(0.2)(self.conv2(out))
        out = nn.LeakyReLU(0.2)(self.conv3(out))
        out = nn.LeakyReLU(0.2)(self.conv4(out))

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
