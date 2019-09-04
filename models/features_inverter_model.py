import torch
import itertools
from collections import OrderedDict
from util.image_pool import ImagePool
import util.util as util
from .base_model import BaseModel
from . import networks
from models.features_extractor import define_Vgg19
import numpy as np
import torch.autograd as autograd
import os
import torch.nn as nn


class FeaturesInverterModel(BaseModel):
    def name(self):
        return 'FeaturesInverterModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--weight_GAN', type=float, default=0.01, help='GAN weight')
            parser.add_argument('--weight_P', type=float, default=0.01, help='P weight')

        parser.add_argument('--layer_num', type=int, default=4, help='weight for classification loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['recon', 'P', 'GAN', 'D']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real', 'fake']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.num_channels = self.opt.input_nc

        channels = [64, 128, 256, 512, 512]
        self.netG = networks.define_features_inverter(channels[opt.layer_num], type="multi_old", init_type=opt.init_type,
                                           init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        self.layer_num = opt.layer_num

        with torch.no_grad():
            self.netFeatures = define_Vgg19(gpu_ids=self.gpu_ids)

        if self.isTrain:
            use_sigmoid = False
        #
            self.netD = networks.define_D_img(3, 64, 'basic', 3, opt.norm, use_sigmoid, opt.init_type,
                                          opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.prec_weights = torch.tensor([2, 1, 0.5, 0.5, 0.2], dtype=torch.float32, device=self.device)
            self.prec_weights = self.prec_weights / torch.sum(self.prec_weights)

            self.fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode='lsgan').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real = input['A'].to(self.device)  # get image data A
        self.real_features = self.netFeatures(self.real).detach() # Extract latent features
        self.image_paths = input['A_paths']  # get image paths

    def output_networks(self, dir_path):
        save_filename = "invert_net_" + str(self.opt.layer_num) + ".pth"
        save_path = os.path.join(dir_path, save_filename)
        print("outputing", save_path)
        net = self.netG

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def forward(self):
        self.fake = self.netG(self.real_features)

    def backward_G(self):
        relu = nn.ReLU()

        self.loss_GAN = self.criterionGAN(self.netD(self.fake), True) * self.opt.weight_GAN

        self.loss_recon = self.criterionL1(self.fake, self.real)

        self.loss_P = 0

        self.fake_layers = self.netFeatures.forward_all_layers(self.fake)
        self.real_layers = self.netFeatures.forward_all_layers(self.real)

        for i in range(5):
            self.loss_P += self.prec_weights[i] * self.criterionMSE(relu(self.fake_layers[i]),
                                                                    relu(self.real_layers[i].detach())) \
                           * self.opt.weight_P

        self.loss_T = self.loss_recon + self.loss_P + self.loss_GAN
        self.loss_T.backward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = loss_D_real + loss_D_fake

        # backward
        loss_D.backward()

        return loss_D

    def backward_D(self):
        fake = self.fake_pool.query(self.fake)
        self.loss_D = self.backward_D_basic(self.netD, self.real, fake)

    def optimize_parameters(self, args):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

