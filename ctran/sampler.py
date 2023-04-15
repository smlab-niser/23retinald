import torch
from torch.utils.data import Sampler

class WeightedRandomSampler(Sampler):
    
    def __init__(self, labels, num_samples=None):                                                                                                              
        self.labels = torch.tensor(labels)                                                                        # labels stores the label matrix
        self.num_samples = len(self.labels) if num_samples is None else num_samples                               # stores the number of samples
        class_sample_count = torch.tensor([(self.labels[:,i] == 1).sum() for i in range(self.labels.shape[1])])   # Finds the of positive samples of each label and stores it in a tensor
        weights = 1.0 / class_sample_count.float()                                                                # Finds weights of each label
        self.samples_weight = (self.labels * weights).sum(dim=1)                                                  # Find the weight for each sample by dot producting the label tensor for that sample
                                                                                                                  # with the counts tensor. The weight would be more if the sample belongs to more minority classes
    # Return the multinomial with sample weight and sample
    def __iter__(self):
        return iter(torch.multinomial(self.samples_weight, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples