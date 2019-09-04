import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NGPU = 1 # Number of GPUs available. Use 0 for CPU mode.
BATCH_SIZE = 64
IMG_SIZE = 112


def define_Vgg19(gpu_ids):
    use_gpu = len(gpu_ids) > 0
    vgg19net = vgg19(models.vgg19(pretrained=True), gpu_ids)

    if use_gpu:
        assert(torch.cuda.is_available())
        vgg19net.cuda(gpu_ids[0])
    return vgg19net

class vgg19(nn.Module):
    def __init__(self, basic_model, gpu_ids=[]):
        super(vgg19, self).__init__()
        basic_model = models.vgg19(pretrained=True)
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.layer_1 = self.make_layers(basic_model,0,2)
        self.layer_2 = self.make_layers(basic_model,2,7)
        self.layer_3 = self.make_layers(basic_model,7,12)
        self.layer_4 = self.make_layers(basic_model,12,21)
        self.layer_5 = self.make_layers(basic_model,21,30)
        self.layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]
        self.set_mean_std()

    def make_layers(self, basic_model, start_layer, end_layer):
        layer = []
        features = next(basic_model.children())
        original_layer_number = 0
        for module in features.children():
            if original_layer_number >= start_layer and original_layer_number < end_layer:
                layer += [module]
            original_layer_number += 1
        return nn.Sequential(*layer)

    def set_mean_std(self):
        mean = torch.zeros(1,3,1,1).to(self.device)
        mean[0,:,0,0] = torch.FloatTensor([0.485, 0.456, 0.406]).to(self.device)
        self.mean = mean

        stdv = torch.zeros(1,3,1,1).to(self.device)
        stdv[0,:,0,0] = torch.FloatTensor([0.229, 0.224, 0.225]).to(self.device)
        self.stdv = stdv

    def pre_process(self, input, min_max=(-1.,1.)):
        mean = self.mean.expand_as(input)
        stdv = self.stdv.expand_as(input)

        return torch.div((((input - min_max[0]) / (min_max[1] - min_max[0])) - mean), stdv)

    def post_process(self, input):
        return input
        return 2 * (input - torch.min(input, dim=1, keepdim=True)[0]) / (torch.max(input, dim=1, keepdim=True)[0] - torch.min(input, dim=1, keepdim=True)[0]) - 1.

    def forward(self, input, from_layer=0, to_layer=5):
        layer_i_output = layer_i_input = input
        if from_layer == 0:
            layer_i_output = layer_i_input = self.pre_process(input)
        else:
            layer_i_output = layer_i_input = input
        for i in range(from_layer, to_layer):
            layer_i = self.layers[i]
            layer_i_output = layer_i(layer_i_input)
            layer_i_input = layer_i_output

        return self.post_process(layer_i_output)

    def forward_all_layers(self, input, from_layer=0, to_layer=5):
        layer_i_output = layer_i_input = input
        layers_output = []
        if from_layer == 0:
            layer_i_output = layer_i_input = self.pre_process(input)
        else:
            layer_i_output = layer_i_input = input
        for i in range(from_layer, to_layer):
            layer_i = self.layers[i]
            layer_i_output = layer_i(layer_i_input)
            layers_output.append(layer_i_output)
            layer_i_input = layer_i_output

        return layers_output

def get_VGG_features(batch, layer=4):
    out = batch
    for i in range(layer):
        out = vgg.layers[i].forward(out)
    return out