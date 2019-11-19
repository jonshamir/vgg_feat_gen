import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFile
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AugmentedDataset(BaseDataset):
    """Dataset class that extracts VGG deep features"""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))

        self.transform = transforms.Compose([
            transforms.ColorJitter(0, 0, 0.2, 0.05),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(10, scale=(0.7, 1), shear=5, resample=2, fillcolor=(255,255,255)),
            transforms.Resize(opt.load_size),
            transforms.RandomCrop(opt.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])

    def __getitem__(self, index):
        """Return a data point and its metadata information"""
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)


