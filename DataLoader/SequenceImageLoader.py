import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging


class SequenceImageLoader(object):
    default_config = {
        "root_path": "/home/zxm/Pictures/Webcam",
        "start": 0,
        "format": "jpg"
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Sequence image loader config: ")
        logging.info(self.config)

        self.img_id = self.config["start"]
        self.img_N = len(glob.glob(pathname=self.config["root_path"] + "/*." + self.config["format"]))

    def __getitem__(self, item):
        file_name = self.config["root_path"] + "/" + str(item) + "." + self.config["format"]
        img = cv2.imread(file_name)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < self.img_N:
            img = self.__getitem__(self.img_id)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]
