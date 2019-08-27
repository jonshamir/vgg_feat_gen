import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NGPU = 1 # Number of GPUs available. Use 0 for CPU mode.
BATCH_SIZE = 64
IMG_SIZE = 112

class vgg19(nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()
        basic_model = models.vgg19(pretrained=True)
        self.layer_1 = self.make_layers(basic_model,0,1)
        self.layer_2 = self.make_layers(basic_model,1,6)
        self.layer_3 = self.make_layers(basic_model,6,11)
        self.layer_4 = self.make_layers(basic_model,11,20)
        self.layer_5 = self.make_layers(basic_model,20,29)
        self.layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]

        self.Tensor = torch.cuda.FloatTensor if NGPU else torch.Tensor
        self.input = self.Tensor(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)

    def make_layers(self, basic_model, start_layer, end_layer):
        layer = []
        features = next(basic_model.children())
        original_layer_number = 0
        for module in features.children():
            if original_layer_number >= start_layer and original_layer_number < end_layer:
                layer += [module]
            original_layer_number += 1
        return nn.Sequential(*layer)

vgg = vgg19().to(DEVICE)

def get_VGG_features(batch, layer=4):
    out = batch
    for i in range(layer):
        out = vgg.layers[i].forward(out)
    return out
