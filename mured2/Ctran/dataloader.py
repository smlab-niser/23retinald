import torch
from torch.utils.data import DataLoader
from data import RetinaDataset
from sampler import WeightedRandomSampler
from transform import Transform

def create_dataloader(data_dir, batch_size, num_workers, size, phase):
    transform = Transform(size=size, phase=phase)

    dataset = RetinaDataset(data_dir=data_dir, split=phase, transform=transform)

    if phase == 'train':
        sampler = WeightedRandomSampler(labels=dataset.labels)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader