import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor


class Encoder:
    def __init__(self, input_size):
        self.conv1 = nn.Conv2D(input_size, )