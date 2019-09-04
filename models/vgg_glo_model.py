import torch
from .base_model import BaseModel
from . import networks
from models.features_extractor import define_Vgg19
import itertools

class VggGLOModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='singleidx')
        parser.set_defaults(gan_mode='GLO')
        parser.set_defaults(ndf=512)

        parser.add_argument('--nz', type=int, default='512', help='Size of the noise')
        parser.add_argument('--inverter_path', type=str, default='pretrained_models/inverter_net_4_zebra_relu.pth',
                            help='path to saved inverter')

        if is_train:
            # parser.add_argument('--inverter_path', type=str, default='pretrained_models//inverter_net_4_zebra.pth', help='path to saved inverter')

            parser.add_argument('--lr_z', type=float, default='0.1', help='Size of the noise')
            parser.add_argument('--lr_g', type=float, default='0.01', help='Size of the noise')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.nz = opt.nz
        self.loss_names = ['G']
        self.visual_names = ['real', 'real_recon', 'fake']
        self.model_names = ['G']

        self.visual_names += ['fake_random', 'interp_0', 'interp_1', 'interp_2', 'interp_3', 'interp_4', 'interp_5']
        self.netG = networks.define_G(opt.nz // 16, opt.netG, gpu_ids=self.gpu_ids)

        self.latent_code = self.generate_latent(opt.dataset_size)

        with torch.no_grad():
            self.netFeatures = define_Vgg19(gpu_ids=self.gpu_ids)
            self.netInverter = networks.define_features_inverter(512, type="multi_old", load_path=opt.inverter_path, gpu_ids=self.gpu_ids)

            channels = [64, 128, 256, 512, 512]

        if self.isTrain:  # only defined during training time
            self.criterionRecon = torch.nn.L1Loss()
            params = itertools.chain(self.netG.parameters(), self.latent_code)

            self.optimizer = torch.optim.Adam([{'params': self.netG.parameters(), 'lr': opt.lr_g},
                                              {'params': self.latent_code, 'lr': opt.lr_z}],
                                            betas=(opt.beta1, 0.999))


            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        self.real = input['A'].to(self.device)  # get image data A
        self.noise = self.latent_code[input['ind'].to(self.device)]
        self.real_features = self.netFeatures(self.real).detach() # Extract latent features
        self.real_recon = self.netInverter(self.real_features.detach()) # Invert features to image
        self.image_paths = input['A_paths' ]  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_features = self.netG(self.noise)  # generate output features from noise input
        self.fake = self.netInverter(self.fake_features.detach()) # Invert features to image

    def generate_latent(self, num_samples):
        z = torch.randn(num_samples, self.nz, device=self.device)
        z /= torch.max(torch.norm(z, dim=1, keepdim=True), torch.ones(num_samples, 1, device=self.device))
        return z.view(-1, self.nz // 16, 4, 4)

    def compute_visuals(self):
        z = torch.randn(1, self.nz, device=self.device)
        z = z.view(-1, self.nz // 16, 4, 4)
        with torch.no_grad():
            self.fake_random = self.netInverter(self.netG(z).detach())

            z1 = self.noise[0].unsqueeze(0)
            z2 = self.noise[-1].unsqueeze(0)
            alphas = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
            for i, alpha in enumerate(alphas):
                z_ = z1 * alpha + z2 * (1-alpha)
                intrep_ = self.netInverter(self.netG(z_).detach())
                setattr(self, 'interp_' + str(i), intrep_)

    def backward(self):
        # print("real", torch.min(self.real_features), torch.max(self.real_features))
        # print("fake", torch.min(self.fake_features), torch.max(self.fake_features))
        self.loss_G = self.criterionRecon(self.fake_features, self.real_features)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def save_networks(self, epoch):
        super(VggGLOModel).save_networks(epoch)
        torch.save()
