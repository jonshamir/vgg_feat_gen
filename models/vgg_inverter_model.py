import torch
from .base_model import BaseModel
from . import networks
from .vgg_extractor import get_VGG_features
from .net_architectures import VGGInverterG, VGGInverterD


class VGGInverterModel(BaseModel):
    @property
    def name(self): return 'VGGInverterModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--vgg_relu', type=bool, default='False', help='Get VGG features after ReLU activation or before')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.vgg_relu = opt.vgg_relu

        # specify the training losses you want to print out
        self.loss_names = ['D', 'G', 'adv', 'feat', 'img']
        # specify the images you want to save and display
        self.visual_names = []
        self.num_preview_images = 6
        for i in range(self.num_preview_images):
            self.visual_names.append('original_image{}'.format(i))
            self.visual_names.append('reconstructed_image{}'.format(i))
        # specify the models you want to save to the disk
        if self.isTrain: self.model_names = ['G', 'D']
        else: self.model_names = ['G']
        # normalization settings
        self.normalize_data = opt.normalize_data;
        if self.normalize_data:
            self.data_mean = opt.data_mean
            self.data_std = opt.data_std

        # define networks
        self.netG = VGGInverterG().to(self.device)

        if self.isTrain:
            self.netD = VGGInverterD().to(self.device)
            # define loss functions
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
        self.real_data = input['A'].to(self.device)
        self.real_feats = self.get_deep_feats(self.real_data)

        if not hasattr(self, 'fixed_real'):
            self.fixed_real = self.real_data.to(self.device)[:16]
            for i in range(self.num_preview_images):
                setattr(self, 'original_image{}'.format(i), self.fixed_real[i:i+1])
            self.fixed_feats = self.get_deep_feats(self.fixed_real)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>"""
        self.fake_data = self.netG(self.real_feats).detach() # generate output image given the input data_A

    def forward_G(self):
        self.forward()
        self.fake_feats = self.get_deep_feats(self.fake_data)

    def backward_G(self):
        self.loss_adv = self.BCE(self.netD(self.netG(self.real_feats)), self.ones)
        self.loss_feat = self.L2(self.real_feats, self.fake_feats)
        self.loss_img = self.L2(self.real_data, self.fake_data)
        self.loss_G = self.loss_adv + self.loss_feat + self.loss_img
        self.loss_G.backward()

    def backward_D(self):
        real_outputs = self.netD(self.real_data)
        fake_outputs = self.netD(self.fake_data)
        D_real_loss = self.BCE(real_outputs, self.ones)
        D_fake_loss = self.BCE(fake_outputs, self.zeros)
        self.loss_D = D_real_loss + D_fake_loss
        self.loss_D.backward()

    def compute_visuals(self):
        fake = self.netG(self.fixed_feats).detach().cpu()
        for i in range(self.num_preview_images):
            setattr(self, 'reconstructed_image{}'.format(i), fake[i:i + 1])

    def optimize_parameters(self, args):
        """Update network weights; will be called in every training iteration"""
        self.forward_G()
        self.backward_D()
        self.D_opt.step()
        self.netD.zero_grad()
        self.netG.zero_grad()
        self.backward_G()
        self.G_opt.step()

    def get_deep_feats(self, data):
        feats = get_VGG_features(data, relu=self.vgg_relu).detach()
        if self.normalize_data:
            feats = (feats - self.data_mean) / self.data_std
            return torch.clamp(feats, -1., 1.)
        return feats

