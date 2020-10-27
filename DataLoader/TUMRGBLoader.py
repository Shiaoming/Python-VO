import os

import cv2
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

from utils.PinholeCamera import PinholeCamera


class TUMRGBLoader(object):
    default_config = {
        'root_path': '/mnt/dataset_hdd/tumrgbd',
        'scene': "rgbd_dataset_freiburg1_360",
        "start": 0,
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("TUMRGB Dataset config: ")
        logging.info(self.config)

        self.img_id = self.config["start"]
        self.root = self.config['root_path']

        assert (Path(self.root).exists()), f"Dataset root path {self.root} dose not exist!"

        self.imgs = []
        self.poses = []
        self.read_imgs()

        self.img_N = len(self.imgs)

        self.cam = PinholeCamera(640.0, 4801.0, 525, 525, 319.5, 239.5)

    def quaternion_to_rotation_matrix(self, quat):
        """
        Args:
            quat: x,y,z,w

        Returns:
            rot_matrix: 3*3 array
        """
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(quat)
        rot_matrix = r.as_matrix()
        return rot_matrix

    def pose_transform(self, posexyzquat):
        """
        Args:
            posexyzquat: tx,ty,tz,qx,qy,qz,qw

        Returns:
            pose: 4*4 array
        """
        pose = np.identity(4)
        pose[:3, 3] = posexyzquat[:3]
        pose[:3, :3] = self.quaternion_to_rotation_matrix(posexyzquat[3:])
        return pose

    def read_imgs(self):
        root_path = self.root
        scene = self.config['scene']

        scene = scene.strip()

        rgbd_gt_file = Path(root_path) / scene / "rgbd_gt.txt"
        with open(rgbd_gt_file, 'r') as f:
            lines = f.readlines()
            lines = np.array([line.strip().split() for line in lines[2:]])

            # time_stamps = [float(stamp) for stamp in lines[:, 0]]
            rgb_list = lines[:, 1]
            # depth_list = lines[:, 2]
            poses = [self.pose_transform(pose) for pose in lines[:, 3:]]
            imgs = [os.path.join(root_path, scene, img) for img in rgb_list]

            self.imgs += imgs
            self.poses += poses

    def __iter__(self):
        return self

    def __getitem__(self, item):
        return cv2.imread(self.imgs[item])

    def __next__(self):
        if self.img_id < self.__len__():
            img = self.__getitem__(self.img_id)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]

    def get_cur_pose(self):
        return self.poses[self.img_id - 1]


if __name__ == "__main__":
    loader = TUMRGBLoader()

    for img in tqdm(loader):
        cv2.putText(img, "Press any key but Esc to continue, press Esc to exit", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 8)
        cv2.imshow("img", img)
        # press Esc to exit
        if cv2.waitKey(10) == 27:
            break
