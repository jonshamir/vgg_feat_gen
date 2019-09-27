import torch
from .base_model import BaseModel
from . import networks
from .vgg_extractor import get_VGG_features, get_all_VGG_features
from .net_architectures import DeepGenerator, DeepDiscriminator, VGGInverterG, calc_gradient_penalty


class VggGenModel(BaseModel):
    @property
    def name(self): return 'VggGenModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--inverter_path', type=str, default='pretrained_models', help='path to saved inverter')
        parser.add_argument('--nz', type=int, default='128', help='Size of the noise')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.vgg_relu = opt.vgg_relu

        # specify the training losses you want to print out
        self.loss_names = ['D', 'G']
        # specify the images you want to save and display
        self.visual_names = ['real_data', 'fake_data']

        # specify the models you want to save to the disk
        if self.isTrain: self.model_names = ['D', 'G']
        else: self.model_names = ['G']
        # normalization settings
        self.normalize_data = opt.normalize_data
        if self.normalize_data:
            self.data_mean = opt.data_mean
            self.data_std = opt.data_std

        # define networks
        self.nz = opt.nz
        self.netG = DeepGenerator().to(self.device)
        self.netInv = VGGInverterG().to(self.device)
        self.netInv.load_state_dict(torch.load(opt.inverter_path))

        if self.isTrain:
            self.netD = DeepDiscriminator(vgg_layer=5, ndf=512).to(self.device)
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
        self.real_feats = self.get_deep_feats(self.real_data)

    def sample_noise(self):
        return torch.randn(self.real_data.size(0), self.nz, device=self.device)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>"""
        self.noise = self.sample_noise()
        self.fake_feats = self.netG(self.noise)
        self.fake_data = self.netInv(self.fake_feats)

    def backward_G(self):
        z_outputs = self.netD(self.fake_feats)
        # self.loss_G = self.BCE(z_outputs, self.ones)
        self.loss_G = -torch.mean(z_outputs)
        self.loss_G.backward()

    def backward_D(self):
        real_outputs = self.netD(self.real_feats)
        fake_outputs = self.netD(self.fake_feats.detach())
        ## Normal loss
        # D_real_loss = self.BCE(real_outputs, self.ones)
        # D_fake_loss = self.BCE(fake_outputs, self.zeros)
        ## Wasserstein loss
        D_real_loss = -torch.mean(real_outputs)
        D_fake_loss = torch.mean(fake_outputs)
        ## Hinge loss
        # D_real_loss = nn.ReLU()(1.0 - real_outputs).mean()
        # D_fake_loss = nn.ReLU()(1.0 + fake_outputs).mean()
        self.loss_D = D_real_loss + D_fake_loss
        if self.opt.gan_mode == 'wgangp':
            loss_D += calc_gradient_penalty(self.netD, self.real_feats, self.fake_feats, self.device)

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