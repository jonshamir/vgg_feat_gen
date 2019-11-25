import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from .net_architectures import DeepGenerator, DeepEncoder, VGGInverterG


class EncoderModel(BaseModel):
    @property
    def name(self): return 'EncoderModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--gen_path', type=str, default='pretrained_models', help='path to saved G')
        parser.add_argument('--inverter_path', type=str, default='pretrained_models', help='path to saved inverter')
        parser.add_argument('--nz', type=int, default='128', help='Size of the noise')
        parser.add_argument('--feat_layer', type=int, default='5', help='VGG layer to get features from')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out
        self.loss_names = ['E']
        # specify the images you want to save and display
        self.visual_names = ['real_noise_inv', 'fake_noise_inv']

        # specify the models you want to save to the disk
        self.model_names = ['G', 'E']

        # define networks
        self.nz = opt.nz
        self.netE = DeepEncoder(layer=opt.feat_layer).to(self.device)
        self.netG = DeepGenerator(layer=opt.feat_layer).to(self.device)
        self.netG.load_state_dict(torch.load(opt.gen_path))
        self.netInv = VGGInverterG(layer=opt.feat_layer).to(self.device)
        self.netInv.load_state_dict(torch.load(opt.inverter_path))

        if self.isTrain:
            # define loss functions
            self.L1 = torch.nn.L1Loss()
            self.L2 = torch.nn.MSELoss()
            # define and initialize optimizers
            self.E_opt = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.E_opt]
            self.batch_size = opt.batch_size

    def set_input(self, input):
        self.image_paths = input['A_paths']
        self.real_data = input['A'].to(self.device).detach()
        self.real_noise = self.sample_noise()

    def sample_noise(self):
        return torch.randn(self.batch_size, self.nz, device=self.device)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>"""
        self.fake_data = self.netG(self.real_noise)
        self.real_noise_inv = self.netInv(self.fake_data)
        self.fake_noise = self.netE(self.fake_data.detach())
        self.fake_noise_inv = self.netInv(self.netG(self.fake_noise))

    def backward_E(self):
        self.loss_E = self.L1(self.fake_noise, self.real_noise)
        self.loss_E.backward()

    def optimize_parameters(self, args):
        """Update network weights; will be called in every training iteration"""
        self.E_opt.zero_grad()
        self.forward()
        self.backward_E()
        self.E_opt.step()
