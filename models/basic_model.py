import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from .net_architectures import BasicGenerator, BasicDiscriminator


class BasicModel(BaseModel):
    @property
    def name(self): return 'BasicModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out
        self.loss_names = ['D', 'G']
        # specify the images you want to save and display
        self.visual_names = ['real_data', 'fake_data']

        # specify the models you want to save to the disk
        if self.isTrain: self.model_names = ['D', 'G']
        else: self.model_names = ['G']

        # define networks
        self.nz = 100
        self.netG = BasicGenerator().to(self.device)

        if self.isTrain:
            self.netD = BasicDiscriminator().to(self.device)
            # define loss functions
            self.L1 = torch.nn.L1Loss()
            self.L2 = torch.nn.MSELoss()
            self.BCE = torch.nn.BCELoss()
            # define target labels
            self.ones = torch.ones([opt.batch_size, 1]).to(self.device)
            self.zeros = torch.zeros([opt.batch_size, 1]).to(self.device)
            # define and initialize optimizers
            self.G_opt = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.D_opt = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.G_opt, self.D_opt]

    def set_input(self, input):
        self.image_paths = input['A_paths']
        self.real_data = input['A'].to(self.device).detach()

    def sample_noise(self):
        return torch.randn(self.real_data.size(0), self.nz, device=self.device)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>"""
        self.noise = self.sample_noise()
        self.fake_data = self.netG(self.noise)

    def backward_G(self):
        self.noise = self.sample_noise()
        z_outputs = self.netD(self.netG(self.noise))
        # self.loss_G = self.BCE(z_outputs, self.ones)
        self.loss_G = -torch.mean(z_outputs)
        self.loss_G.backward()

    def backward_D(self):
        real_outputs = self.netD(self.real_data)
        fake_outputs = self.netD(self.fake_data)
        # # Normal loss
        # D_real_loss = self.BCE(real_outputs, self.ones)
        # D_fake_loss = self.BCE(fake_outputs, self.zeros)
        ## Wasserstein loss
        D_real_loss = -torch.mean(real_outputs)
        D_fake_loss = torch.mean(fake_outputs)
        ## Hinge loss
        # D_real_loss = nn.ReLU()(1.0 - real_outputs).mean()
        # D_fake_loss = nn.ReLU()(1.0 + fake_outputs).mean()
        self.loss_D = D_real_loss + D_fake_loss
        self.loss_D.backward()

    def optimize_parameters(self, args):
        """Update network weights; will be called in every training iteration"""
        self.D_opt.zero_grad()
        self.forward()
        self.backward_D()
        self.D_opt.step()

        self.D_opt.zero_grad()
        self.G_opt.zero_grad()
        self.backward_G()
        self.G_opt.step()

