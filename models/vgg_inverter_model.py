import torch
from .base_model import BaseModel
from . import networks
from .vgg_extractor import get_VGG_features, get_all_VGG_features
from .net_architectures import VGGInverterG, VGGInverterD, VGGInverterDSpectral


class VGGInverterModel(BaseModel):
    @property
    def name(self): return 'VGGInverterModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.vgg_relu = opt.vgg_relu

        # specify the training losses you want to print out
        self.loss_names = ['D', 'G', 'adv', 'feat', 'img']
        # specify the images you want to save and display
        self.visual_names = ['real_data', 'fake_data']

        # specify the models you want to save to the disk
        if self.isTrain: self.model_names = ['G', 'D']
        else: self.model_names = ['G']
        # normalization settings
        self.normalize_data = opt.normalize_data
        if self.normalize_data:
            self.data_mean = opt.data_mean
            self.data_std = opt.data_std

        # define networks
        self.netG = VGGInverterG().to(self.device)

        if self.isTrain:
            self.netD = VGGInverterD().to(self.device)
            # define loss functions
            self.L1 = torch.nn.L1Loss()
            self.L2 = torch.nn.MSELoss()
            self.BCE = torch.nn.BCELoss()
            # define target labels
            self.ones = torch.ones([opt.batch_size, 1]).to(self.device)
            self.zeros = torch.zeros([opt.batch_size, 1]).to(self.device)
            # define and initialize optimizers
            self.G_opt = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.D_opt = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.G_opt, self.D_opt]

    def set_input(self, input):
        self.image_paths = input['A_paths']
        self.real_data = input['A'].to(self.device).detach()
        self.real_feats = self.get_deep_feats(self.real_data)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>"""
        self.fake_data = self.netG(self.real_feats)
        if self.isTrain:
            self.fake_feats = self.get_deep_feats(self.fake_data)

    def backward_G(self):
        self.loss_adv = self.BCE(self.netD(self.netG(self.real_feats)), self.ones) * 0.1
        self.loss_feat = self.calc_loss_feat(self.fake_data, self.real_data)
        self.loss_img = self.L2(self.fake_data, self.real_data)
        self.loss_G = self.loss_adv + self.loss_feat + self.loss_img
        self.loss_G.backward()

    def backward_D(self):
        real_outputs = self.netD(self.real_data)
        fake_outputs = self.netD(self.fake_data.detach())
        D_real_loss = self.BCE(real_outputs, self.ones)
        D_fake_loss = self.BCE(fake_outputs, self.zeros)
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

    def get_deep_feats(self, data):
        feats = get_VGG_features(data, relu=self.vgg_relu)
        if self.normalize_data:
            feats = (feats - self.data_mean) / self.data_std
            return torch.clamp(feats, -1., 1.)
        return feats

    def calc_loss_feat(self, fake_data, real_data):
        loss_feat = 0
        weights = torch.tensor([2.0, 1.0, 0.5, 0.5, 0.2]).to(self.device)
        weights = weights / torch.sum(weights)
        all_feats_real = get_all_VGG_features(real_data, relu=True)
        all_feats_fake = get_all_VGG_features(fake_data, relu=True)
        for i in range(5):
            loss_feat += weights[i] * self.L2(all_feats_fake[i],  all_feats_real[i].detach())

        return loss_feat


