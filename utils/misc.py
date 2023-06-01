
import torch.nn as nn

__all__ = ['AverageMeter', 'GroupNormCreator']


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GroupNormCreator:
    # Functor for creating GN layer
    def __init__(self, num_groups=32, min_num_channels_per_group=4):
        self.num_groups = num_groups
        self.min_num_channels_per_group = min_num_channels_per_group

    def __call__(self, num_features):
        return nn.GroupNorm(num_channels=num_features, num_groups=min(self.num_groups,
                                                                      num_features // self.min_num_channels_per_group))
