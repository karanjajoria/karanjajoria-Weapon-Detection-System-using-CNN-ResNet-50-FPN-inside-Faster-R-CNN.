import torchvision.transforms as T
from PIL import Image
import random
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class ToTensor:
    def __call__(self, image):
        if isinstance(image, Image.Image):
            return T.ToTensor()(image)
        return image

class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image):
        if random.random() < self.prob:
            return T.functional.hflip(image)
        return image
