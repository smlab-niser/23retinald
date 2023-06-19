import torch
from torch.utils.data import Sampler
from skmultilearn.problem_transform import LabelPowerset
import random
import numpy as np
import math
from collections import defaultdict


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
    
    
# class LPRandomOverSampler(Sampler):
#     def __init__(self, labels, num_samples=None, sample_percent=0.1):
#         self.labels = torch.tensor(labels)
#         self.num_samples = len(self.labels) if num_samples is None else num_samples
#         self.sample_percent = sample_percent
#         self.class_sample_count = torch.tensor([(self.labels[:,i] == 1).sum() for i in range(self.labels.shape[1])])
#         self.class_probabilities = self.class_sample_count / self.class_sample_count.sum()
        
        
#     def __iter__(self):
#         lp = LabelPowerset()
#         print(self.labels.shape)
#         lp.fit(self.labels)
#         indices = []
#         for sample_idx in range(self.num_samples):
#             label = self.labels[sample_idx]
#             label_powerset = lp.transform(label.reshape(1, -1))
#             label_set = set(lp.inverse_transform(label_powerset)[0])
#             sample_weight = self.class_probabilities[list(label_set)].prod()
#             if random.random() < self.sample_percent * sample_weight:
#                 indices.append(sample_idx)
#         return iter(indices)
    
#     def __len__(self):
#         return len(self.labels)
    
    
class LPRandomOverSampler(Sampler):
    def __init__(self, labels, num_samples=None, sample_percent=0.1):
        self.labels = torch.tensor(labels)
        self.num_samples = len(self.labels) if num_samples is None else num_samples
        self.sample_percent = sample_percent
        
        labels = np.array(self.labels)
        print(labels, "\n", labels.shape)

        labelpowerset = LabelPowerset()
        labelsets = np.array(labelpowerset.transform(labels))
        print(labelsets, "\n", labelsets.shape)
        
        label_set_bags = defaultdict(list)

        for idx, label in enumerate(labelsets):
            label_list = [label]
            label_set_bags[tuple(label_list)].append(idx)

        mean_size = 0
        for label, samples in label_set_bags.items():
            mean_size += len(samples)

        # ceiling
        mean_size = math.ceil(mean_size / len(label_set_bags))

        minority_bag = []
        for label, samples in label_set_bags.items():
            if len(samples) < mean_size:
                minority_bag.append(label)

        if len(minority_bag) == 0:
            print('There are no labels below the mean size. mean_size: ', mean_size)
            self.indices = np.arange(len(self.labels))
            return

        mean_increase = self.num_samples // len(minority_bag)

        def custom_sort(label):
            return len(label_set_bags[label])

        minority_bag.sort(reverse=True, key=custom_sort)
        acc_remainders = np.zeros(len(minority_bag), dtype=np.int32)
        clone_samples = []

        for idx, label in enumerate(minority_bag):
            increase_bag = min(mean_size - len(label_set_bags[label]), mean_increase)

            remainder = mean_increase - increase_bag

            if remainder == 0:
                extra_increase = min(mean_size - len(label_set_bags[label]) - increase_bag, acc_remainders[idx])
                increase_bag += extra_increase
                remainder = acc_remainders[idx] - extra_increase

            self._distribute_remainder(remainder, acc_remainders, idx + 1)

            for i in range(increase_bag):
                x = random.randint(0, len(label_set_bags[label]) - 1)
                clone_samples.append(label_set_bags[label][x])

        self.indices = np.concatenate((np.arange(len(self.labels)), np.array(clone_samples)))

    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
    def _distribute_remainder(self, r, r_dist, idx):
        p = len(r_dist) - idx + 1
        value = r // p
        curr_rem = r % p

        r_dist[idx:] = np.add(r_dist[idx:], value)

        if curr_rem > 0:
            start = len(r_dist) - curr_rem
            r_dist[start:] = np.add(r_dist[start:], 1)