import torch
import torch.nn as nn
import torchvision.utils as vutils
from pathlib import Path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    """
        Convolutional Generator
    """
    def __init__(self, nc=3):
        super(Generator, self).__init__()
        self.conv = nn.Sequential(
            # 7 -> 7
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7 -> 14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 14 -> 28
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 28 -> 56
            nn.ConvTranspose2d(64, nc, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        # input: (N, 100)
        out = self.conv(input)
        return out

G = Generator().to(DEVICE)
G.load_state_dict(torch.load(str(Path().absolute()) + '/inverter_frogs.pkl'))
G.eval()

def features2images(feats):
    fake = G(feats).detach().cpu()
    img = vutils.make_grid(fake, normalize=True, pad_value=1)
    return img.permute(1, 2, 0) # reorganize image channel order
