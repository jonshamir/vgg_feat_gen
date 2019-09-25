import torch
from .base_model import BaseModel
from . import networks
from models.vgg_extractor import get_VGG_features

class VggGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(gan_mode='wgangp')
        parser.set_defaults(ndf=512)
        parser.add_argument('--inverter_path', type=str, default='pretrained_models//inverter_net_4_zebra_relu.pth',
                            help='path to saved inverter')
        parser.add_argument('--nz', type=int, default='512', help='Size of the noise')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.nz = opt.nz
        self.vgg_relu = opt.vgg_relu

        # specify the training losses you want to print out
        self.loss_names = ['G', 'D']
        # specify the images you want to save and display
        visual_names = ['real_data', 'fake_data']
        visual_names_random = []
        for i in range(8): visual_names_random += ['fake_' + str(i)]
        visual_names_interp = []
        for i in range(6): visual_names_interp += ['interp_' + str(i)]

        self.visual_names += [visual_names, visual_names_random, visual_names_interp]

        # specify the models you want to save to the disk
        if self.isTrain: self.model_names = ['G', 'D']
        else: self.model_names = ['G']
        # normalization settings
        self.normalize_data = opt.normalize_data;
        if self.normalize_data:
            self.data_mean = opt.data_mean
            self.data_std = opt.data_std

        # define networks
        self.netG = networks.define_G(opt.nz, opt.netG, gpu_ids=self.gpu_ids, relu_out=opt.vgg_relu, tanh_out=opt.normalize_data)

        with torch.no_grad():
            self.netInverter = networks.define_features_inverter(opt.input_nc, type='new',
                                                                 load_path=opt.inverter_path, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(512, opt.ndf, opt.netD, gpu_ids=self.gpu_ids)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # define and initialize optimizers
            self.G_opt = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.D_opt = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.G_opt, self.D_opt]

    def set_input(self, input):
        self.real_data = input['A'].to(self.device)
        self.real_feats = self.get_deep_feats(self.real_data)

    def forward(self, noise=None):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if noise is None:
            noise = self.sample_noise(self.opt.nz)
        self.noise = noise
        self.fake_feats = self.netG(self.noise)  # generate output features from noise input
        self.fake_data = self.netInverter(self.fake_feats) # Invert features to image
        self.fake_2 = self.fake_data[-1].unsqueeze(0)  # Invert features to image

    def sample_noise(self, nz):
        return torch.randn(self.real_data.size(0), self.nz, device=self.device)

    def backward_G(self):
        # print("real", torch.min(self.real_feats), torch.max(self.real_feats))
        # print("fake", torch.min(self.fake_feats), torch.max(self.fake_feats))
        self.loss_G = self.criterionGAN(self.netD(self.fake_feats), True)
        self.loss_G.backward()

    def backward_D(self):
        self.loss_D = self.backward_D_basic(self.netD, self.real_feats, self.fake_feats)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        if self.opt.gan_mode == 'wgangp':
            #wgan-gp
            gradient_penalty, gradients = networks.cal_gradient_penalty(netD, real, fake, self.device)
            # Combined loss and calculate gradients
            loss_D += gradient_penalty

        loss_D.backward()
        return loss_D

    def optimize_parameters(self, optimize_G=True):
        self.forward()
        if optimize_G:
            self.set_requires_grad([self.netD], False)
            self.G_opt.zero_grad()
            self.backward_G()
            self.G_opt.step()
        self.set_requires_grad([self.netD], True)
        self.D_opt.zero_grad()
        self.backward_D()
        self.D_opt.step()

    def compute_visuals(self):
        self.netG.eval()
        with torch.no_grad():

            for i in range(8):
                z = self.sample_noise(self.opt.nz)
                fake = self.netInverter(self.netG(z).detach())
                setattr(self, 'fake_' + str(i), fake)

            z1 = self.noise[0].unsqueeze(0)
            z2 = self.noise[-1].unsqueeze(0)
            alphas = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
            for i, alpha in enumerate(alphas):
                z_ = z1 * alpha + z2 * (1-alpha)
                intrep_ = self.netInverter(self.netG(z_).detach())
                setattr(self, 'interp_' + str(i), intrep_)
        self.netG.train()

    def get_deep_feats(self, data):
        feats = get_VGG_features(data, relu=self.vgg_relu).detach()
        if self.normalize_data:
            feats = (feats - self.data_mean) / self.data_std
            return torch.clamp(feats, -1., 1.)
        return feats