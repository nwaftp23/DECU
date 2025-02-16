import os
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class ImagePaths(Dataset):
    def __init__(self, paths, train=False, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        if not train:
            self.labels["file_path_"] = paths
            self._length = len(paths)
        else:
            self.hdf5 = paths
            #self.labels["file_path_"] = [os.path.join('/home/phoenix/imagenet/ILSVRC2012_train/data', p) for p 
            #    in self.labels['relpath']]
            self.labels["file_path_"] = [p for p in self.labels['relpath']]
            self._length = len(self.labels['relpath'])
        self.train = train
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        if self.train:
            image = np.array(self.hdf5[image_path])
        else:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
