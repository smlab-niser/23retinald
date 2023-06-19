import torch
import torchvision.transforms as transforms

class Transform:
    def __init__(self, size, phase):
        self.phase = phase
        print(size)
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }

    def __call__(self, sample):
        
        # Check if the image has 4 channels
        if sample.mode == 'RGBA':
            # Convert RGBA to RGB
            sample = sample.convert('RGB')
            
        # Resize the image
        w, h = sample.size
        aspect_ratio = w / h
        new_h = sample.size[1]
        new_w = int(new_h * aspect_ratio)
        resized_image = sample.resize((new_w, new_h))

        # Apply the other transforms
        transformed_image = self.data_transforms[self.phase](resized_image)

        return transformed_image