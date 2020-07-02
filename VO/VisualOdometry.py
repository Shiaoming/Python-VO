# based on: https://github.com/uoip/monoVO-python

import numpy as np
import cv2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    """
    A simple frame by frame visual odometry
    """

    def __init__(self, detector, matcher, cam):
        """
        :param detector: a feature detector can detect keypoints their descriptors
        :param matcher: a keypoints matcher matching keypoints between two frames
        :param cam: camera parameters
        """
        # feature detector and keypoints matcher
        self.detector = detector
        self.matcher = matcher

        # camera parameters
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        # frame index counter
        self.index = 0

        # keypoints and descriptors
        self.kpts = {}
        self.desc = {}

        # pose of current frame
        self.cur_R = None
        self.cur_t = None

    def update(self, image, absolute_scale):
        """
        update a new image to visual odometry, and compute the pose
        :param image: input image
        :param absolute_scale: the absolute scale between current frame and last frame
        :return: R and t of current frame
        """
        kpts, desc = self.detector(image)

        # first frame
        if self.index == 0:
            # save keypoints and descriptors
            self.kpts['ref'] = kpts
            self.desc['ref'] = desc

            # start point
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3, 1))
            return
        else:
            # update keypoints and descriptors
            self.kpts['cur'] = kpts
            self.desc['cur'] = desc

            # match keypoints
            px_ref, px_cur = self.matcher(self.kpts, self.desc)

            # compute relative R,t between ref and cur frame
            E, mask = cv2.findEssentialMat(px_cur, px_ref, focal=self.focal, pp=self.pp,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, px_cur, px_ref, focal=self.focal, pp=self.pp)

            # get absolute pose based on absolute_scale
            if (absolute_scale > 0.1):
                self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)

            # save keypoints and descriptors
            self.kpts['ref'] = kpts
            self.desc['ref'] = desc

        return self.cur_R, self.cur_t
