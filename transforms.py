from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import random


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor

class Transforms:
    def __init__(self, use_ir: bool, cfg, dataset_path: str = None, image_size: int = 224):
        self.use_ir = use_ir
        mean, std = self.compute_ds_stats(dataset_path, image_size=image_size)
              
        class Stats: 
            pass
        self.stats = Stats()
        self.stats.mean = mean.tolist()
        self.stats.std = std.tolist()
        
        if use_ir:
            self.global_transform, self.local_transform  = self.ir_transform(self.stats, cfg)
        else:
            self.global_transform, self.local_transform = self.rgb_transform(self.stats, cfg)
    
    def ir_transform(self, stats, cfg):
        normalize = transforms.Normalize(
                stats.mean, # 1 channel
                stats.std
            )
        
        def ir_augs(scale):
            
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    cfg.image_size,
                    scale=scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=.5),
                transforms.RandomRotation(degrees=5),
                transforms.Grayscale(num_output_channels=1),     
                  
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5), sigma=(0.1, 2.0))], p=0.3
                ),
                
                transforms.RandomApply(
                    [transforms.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.3,
                ),
                
                transforms.RandomApply(
                    [transforms.RandomAutocontrast()], p=0.2,
                ),
                
                transforms.ToTensor(),  
                AddGaussianNoise(mean=0.0, std=0.02, p=0.5),               
                normalize,
            ])
        
        global_transform = ir_augs(cfg.global_crop_scale)
        local_transform = ir_augs(cfg.local_crop_scale)
        
        return global_transform, local_transform
    
    def rgb_transform(self, stats, cfg):
        
        normalize = transforms.Normalize(
            stats.mean,  # 3 channels
            stats.std
        )
        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )
        
        def rgb_augs(scale):
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    cfg.image_size,
                    scale=scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                transforms.ToTensor(),
                normalize
            ])
        
        global_transform = rgb_augs(cfg.global_crop_scale)
        local_transform  = rgb_augs(cfg.local_crop_scale)
        
        return global_transform, local_transform
    
    def compute_ds_stats(self, dataset_path, image_size):
        
        if dataset_path is None:
            return torch.tensor([0.5]), torch.tensor([0.25])
        
        if self.use_ir:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        
        dataset = datasets.ImageFolder(
            root=dataset_path,
            transform=transform
        )
        
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        
        mean = 0.0
        std = 0.0
        total_images_count = 0

        for images, _ in tqdm(loader):
            images = images.view(images.size(0), images.size(1), -1) 
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images_count += images.size(0)

        mean /= total_images_count
        std /= total_images_count
        return mean, std