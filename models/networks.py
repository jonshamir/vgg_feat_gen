import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
from .net_architectures import VGGInverterG, VGGInverterD

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type == 'group':
        norm_layer = lambda x: nn.GroupNorm(32, x)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    net = net_to_device(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def net_to_device(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net

def define_G(nz, netG='basic', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], relu_out=False, tanh_out=False):
    net = None
    norm_layer = get_norm_layer(norm_type='group')

    if netG == 'basic':
        net = DeepGenerator(nz, norm_layer)
    if netG == 'advanced':
        net = DeepGenerator2(nz, gen_layer=5, ngf=128, norm_layer=norm_layer, relu_out=relu_out, tanh_out=tanh_out)
    elif netG == 'student':
        net = student_generator()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, use_sigmoid=True, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type='none')

    if netD == 'basic':
        net = DeepDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'advanced':
        net = DeepDiscriminator2(gen_layer=5, ndf=ndf, norm_layer=norm_layer)
    elif netD == 'student':
        net = student_discriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D_img(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

    return init_net(netD, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

# class PerceptualLoss(nn.Module):
#     def __init__(self, gpu_ids):
#         """ Initialize the GANLoss class.
#
#         Parameters:
#             gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
#             target_real_label (bool) - - label for a real image
#             target_fake_label (bool) - - label of a fake image
#
#         Note: Do not use sigmoid as the last layer of Discriminator.
#         LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
#         """
#         super(PerceptualLoss, self).__init__()
#         self.perceptual_model = dm.DistModel()
#         use_gpu = len(gpu_ids) > 0
#         self.perceptual_model.initialize(model='net-lin', net='vgg', use_gpu=use_gpu, spatial=False)
#
#     def __call__(self, real, fake):
#         loss = self.perceptual_model.forward_pair(real, fake)
#         return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class DeepGenerator(nn.Module):
    def __init__(self, nz, norm_layer=nn.GroupNorm):
        super(DeepGenerator, self).__init__()
        self.nz = nz
        model = []

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model += [nn.Conv2d(nz, 64, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(64)]
        model += [nn.ReLU()]

        model += [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(128)]
        model += [nn.ReLU()]

        model += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(256)]
        model += [nn.ReLU()]

        model += [nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        model += [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(512)]
        model += [nn.ReLU()]

        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=use_bias)]
        model += [norm_layer(512)]
        model += [nn.ReLU()]

        model += [nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(512)]
        model += [nn.ReLU()]

        model += [nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=1, bias=use_bias)]
        # model += [norm_layer(512)]
        # model += [nn.Tanh()]
        # model += [nn.ReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print(x.size())
        return self.model(x)

class DeepDiscriminator(nn.Module):
    def __init__(self, in_nc, ndf=512, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(DeepDiscriminator, self).__init__()
        model = []
        fc = []

        in_nc = 512
        #Default i=3: 512x14x14 -> 512x7x7
        model += [nn.Conv2d(int(in_nc), int(ndf),
                            kernel_size=4, stride=2, padding=1, bias=False),
                  nn.LeakyReLU(0.2, True)]


        #Default i=0: 512x7x7 -> 1024x3x3
        #Default i=1: 1024x3x3 -> 2048x1x1
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(int(ndf * mult), int(ndf * 2 * mult),
                                kernel_size=4, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(int(ndf * 2 * mult)),
                         nn.LeakyReLU(0.2, True)]

        #Default i=0: 512 -> 1024
        #Default i=1: 1024 -> 2048
        in_channels = ndf * 2 * mult
        for i in range(2):
            mult = 2 ** i
            fc += [
                nn.Linear(int(in_channels/mult), int(in_channels/ (mult * 2)), bias=False),
                nn.BatchNorm1d(int(in_channels/ (mult * 2))),
                nn.LeakyReLU(True)
                ]

        #Default 128 -> 512
        fc += [ nn.Linear(int(in_channels/ (mult * 2)), 1, bias=False) ]

        if use_sigmoid:
            fc += [nn.Sigmoid()]

        self.fc = nn.Sequential(*fc)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print(x.size())

        L1 = self.model(x)
        # print(L1.size())
        output = self.fc(L1.squeeze())
        # print(output.size())
        return output

class DeepGenerator2(nn.Module):
    # initializers
    def __init__(self, nz, gen_layer=5, ngf=128, norm_layer=nn.BatchNorm2d, relu_out=False, tanh_out=False):
        super(DeepGenerator2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.nz = nz
        gen_all_ch = [3, 64, 128, 256, 512, 512]
        gen_ch = gen_all_ch[gen_layer]
        num_layers = math.ceil(math.sqrt(224 // (2 ** (gen_layer - 1))))

        model = []

        in_ch = nz
        out_ch = ngf
        model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(out_ch)]
        model += [nn.LeakyReLU(0.2, True)]

        for i in range(num_layers):
            in_ch = out_ch
            out_ch = min(in_ch * 2, 512)
            model += [nn.Upsample(scale_factor=2)]
            model += [nn.Conv2d(in_ch, out_ch , kernel_size=3, stride=1, padding=1, bias=use_bias)]
            model += [norm_layer(out_ch)]
            model += [nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(out_ch, out_ch , kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(out_ch)]
        model += [nn.LeakyReLU(0.2, True)]

        padding = 0 if (gen_layer == 5) else 1
        model += [nn.Conv2d(out_ch, gen_ch, kernel_size=3, stride=1, padding=padding, bias=True)]
        if relu_out: model += [nn.ReLU()]
        if tanh_out: model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(self.nz, self.nz)

    # forward method
    def forward(self, input):
        l1 = self.fc(input)
        output = self.model(l1.view(-1, self.nz, 1, 1))

        return output


class DeepDiscriminator2(nn.Module):
    # initializers
    def __init__(self, gen_layer=5, ndf=128, norm_layer=get_norm_layer('none')):
        super(DeepDiscriminator2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        gen_all_ch = [3, 64, 128, 256, 512, 512]
        gen_ch = gen_all_ch[gen_layer]
        feature_size = 224 // (2 ** (gen_layer - 1))
        num_layers = math.ceil(math.sqrt(feature_size))
        num_strided_layers = num_layers - 2

        model = []

        in_ch = gen_ch
        out_ch = ndf
        model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)]
        model += [nn.LeakyReLU(0.2, True)]

        for i in range(2 * num_strided_layers - 1):
            in_ch = out_ch
            out_ch = min(in_ch * 2, 512)
            stride = 2 if ((i % 2) == 0) else 1
            model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=use_bias)]
            model += [norm_layer(out_ch)]
            model += [nn.LeakyReLU(0.2, True)]

        for i in range(num_layers):
            in_ch = out_ch
            out_ch = max(in_ch // 2, 128)
            model += [nn.Conv2d(in_ch, out_ch , kernel_size=3, stride=1, padding=1, bias=use_bias)]
            model += [nn.LeakyReLU(0.2, True)]
            model += [nn.Conv2d(out_ch, out_ch , kernel_size=3, stride=1, padding=1, bias=use_bias)]
            model += [norm_layer(out_ch)]
            model += [nn.LeakyReLU(0.2, True)]

        self.layer_size = 4 * 4 * out_ch

        self.model = nn.Sequential(*model)
        fc = [nn.Linear(self.layer_size, ndf)]
        fc += [nn.Linear(ndf, 1)]

        self.fc = nn.Sequential(*fc)

    # forward method
    def forward(self, input):
        l1 = self.model(input)
        output = self.fc(l1.view(-1, self.layer_size ))

        return output

import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        # print(self.shape, input.shape)
        return input.view(*self.shape)

class student_generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(student_generator, self).__init__()
        self.deconv0 = nn.Conv2d(128, d, 3, 1, 1)
        self.deconv0_bn = nn.BatchNorm2d(d)

        self.pooling1 = nn.Upsample(scale_factor=2)
        self.deconv1 = nn.Conv2d(d, d*2, 3, 1, 1)
        self.deconv1_bn = nn.BatchNorm2d(d*2)

        self.pooling2 = nn.Upsample(scale_factor=2)
        self.deconv2 = nn.Conv2d(d*2, d*4, 3, 1, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)

        self.pooling3 = nn.Upsample(scale_factor=2)
        self.deconv3 = nn.Conv2d(d*4, d*4, 3, 1, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*4)


        self.pooling4 = nn.Upsample(scale_factor=2)
        self.deconv4 = nn.Conv2d(d*4, d*4, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*4)

        self.deconv5 = nn.Conv2d(d*4, d*8, 3, 1, 1)
        self.deconv5_bn = nn.BatchNorm2d(d*8)

        self.deconv6 = nn.Conv2d(d*8, 512, 3, 1, 0)
        self.input = nn.Sequential(View(-1,128), nn.Linear(128,128), View(-1,128,1,1))

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = self.input(input)
        x = F.leaky_relu(self.deconv0_bn(self.deconv0(x)), 0.2)
        x = self.pooling1(x)
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(x)), 0.2)
        x = self.pooling2(x)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = self.pooling3(x)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = self.pooling4(x)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.leaky_relu(self.deconv5_bn(self.deconv5(x)), 0.2)
        x = F.relu(self.deconv6(x))

        return x

class student_discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(student_discriminator, self).__init__()
        self.conv1 = nn.Conv2d(512, d, 3, 1, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)

        self.conv5 = nn.Conv2d(d*8, d*4, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d*4)

        self.conv6 = nn.Conv2d(d*4, d*4, 3, 1, 1)
        self.conv6_bn = nn.BatchNorm2d(d*4)

        self.conv7 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.conv7_bn = nn.BatchNorm2d(d*2)

        self.conv8 = nn.Conv2d(d*2, d*2, 3, 1, 1)
        self.conv8_bn = nn.BatchNorm2d(d*2)

        self.conv9 = nn.Conv2d(d*2, d, 3, 1, 1)
        self.conv9_bn = nn.BatchNorm2d(d)

        self.conv10 = nn.Conv2d(d, 1, 3, 1, 1)
        self.linear1 = nn.Linear(d*9,d)
        self.linear2 = nn.Linear(d,1)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        x = F.leaky_relu(self.conv7(x), 0.2)
        x = F.leaky_relu(self.conv8(x), 0.2)
        x = F.leaky_relu(self.conv9(x), 0.2)
        #x = F.leaky_relu(self.conv10(x), 0.2)

        #x = self.conv10(x)
        x = x.view(x.shape[0],-1)
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = self.linear2(x)
        #x = F.sigmoid(x)

        return x


def define_features_inverter(input_nc, type='new', load_path=None, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if type == 'new':
        net = VGGInverterG()
    else:
        net = FeaturesInverter(input_nc)

    if not load_path is None:
        net.load_state_dict(torch.load(load_path))
        return net_to_device(net, gpu_ids)

    net = init_net(net, init_type, init_gain, gpu_ids)

    return net

class FeaturesInverter(nn.Module):
    def __init__(self, nc=3):
        #Currently only for the deepest layer
        super(FeaturesInverter, self).__init__()
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
        out = self.conv(input)
        return out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
