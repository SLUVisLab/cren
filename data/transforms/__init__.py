from torchvision import transforms

available_transforms: dict = {
    'horizontal_flip': transforms.RandomHorizontalFlip,
    'vertical_flip': transforms.RandomVerticalFlip,
    'resize': transforms.Resize,
    'normalize': transforms.Normalize,
    'random_crop': transforms.RandomCrop,
    'center_crop': transforms.CenterCrop,
}


def get_transforms(commands: dict):
    assert 'normalize' in commands.keys(), "Your data is not being normalized!"
    selected_transforms = []
    for key, parameters in commands.items():
        if key is not 'normalize':
            selected_transforms.append(available_transforms[key](*parameters))
        
    return transforms.Compose([
        *selected_transforms,
        transforms.ToTensor(),
        available_transforms['normalize'](*parameters)
    ])
