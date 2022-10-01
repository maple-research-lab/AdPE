
from PIL import ImageFilter
import random
import numpy as np

class MultiTransform:
    def __init__(
        self,
        transform_train,
        crop_num,
    ):
        self.transform_train = transform_train
        self.crop_num = crop_num
    def __call__(self, x):
        crops = []
        for k in range(self.crop_num):
            q = self.transform_train(x)
            crops.append(q)
        return crops

