from torchvision import transforms

def build_transform(cfg, use_ir):
    if use_ir:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4226], std=[0.1795]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4807, 0.4986, 0.4881],
                std=[0.2233, 0.2059, 0.1738],
            ),
        ])    