import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from .net_architectures import DeepGenerator, DeepEncoder


class EncoderModel(BaseModel):
    @property
    def name(self): return 'EncoderModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--gen_path', type=str, default='pretrained_models', help='path to saved inverter')
        parser.add_argument('--nz', type=int, default='128', help='Size of the noise')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out
        self.loss_names = ['E']
        # specify the images you want to save and display
        self.visual_names = ['real_data']

        # specify the models you want to save to the disk
        self.model_names = ['G', 'E']

        # define networks
        self.nz = opt.nz
        self.netE = DeepEncoder().to(self.device)
        self.netG = DeepGenerator().to(self.device)
        self.netG.load_state_dict(torch.load(opt.gen_path))

        if self.isTrain:
            # define loss functions
            self.L1 = torch.nn.L1Loss()
            self.L2 = torch.nn.MSELoss()
            # define and initialize optimizers
            self.E_opt = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.E_opt]

    def set_input(self, input):
        self.image_paths = input['A_paths']
        self.real_data = input['A'].to(self.device).detach()

    def sample_noise(self):
        return torch.randn(32, self.nz, device=self.device)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>"""
        self.real_noise = self.sample_noise()
        self.fake_data = self.netG(self.real_noise)
        self.fake_noise = self.netE(self.fake_data.detach())

    def backward_E(self):
        self.loss_E = self.L1(self.fake_noise, self.real_noise)
        self.loss_E.backward()

    def optimize_parameters(self, args):
        """Update network weights; will be called in every training iteration"""
        self.E_opt.zero_grad()
        self.forward()
        self.backward_E()
        self.E_opt.step()
