import numpy as np
import cv2
from utils.tools import dict_update


class FrameByFrameMatcher(object):
    default_config = {
        "KNN": {
            "HAMMING": True,  # For ORB Binary descriptor, only can use hamming matching
            "first_N": 300,  # For hamming matching, use first N min matches
        },
        "FLANN": {
            "kdTrees": 5,
            "searchChecks": 50
        },
        "distance_ratio": 0.75
    }

    def __init__(self, matcher_type="FLANN", config=None):
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.matcher_type = matcher_type

        if self.matcher_type == "KNN":
            if self.config["KNN"]["HAMMING"]:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.matcher = cv2.BFMatcher()
        elif self.matcher_type == "FLANN":
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=self.config["FLANN"]["kdTrees"])
            search_params = dict(checks=self.config["FLANN"]["searchChecks"])  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher type: {self.matcher_type}")

    def match(self, desc):
        self.good = []
        if self.matcher_type == "KNN" and self.config["KNN"]["HAMMING"]:
            matches = self.matcher.match(desc["ref"], desc["cur"])
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            # self.good = matches[:self.config["KNN"]["first_N"]]
            for i in range(self.config["KNN"]["first_N"]):
                self.good.append([matches[i]])
        else:
            matches = self.matcher.knnMatch(desc["ref"], desc["cur"], k=2)
            # Apply ratio test
            for m, n in matches:
                if m.distance < self.config["distance_ratio"] * n.distance:
                    self.good.append([m])
        return self.good

    def draw_matches(self, imgs, kpts):
        img = cv2.drawMatchesKnn(imgs["ref"], kpts["ref"],
                                 imgs["cur"], kpts["cur"],
                                 self.good, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img

    def get_good_keypoints(self, kpts):
        kp_ref = np.zeros([len(self.good), 2])
        kp_cur = np.zeros([len(self.good), 2])
        for i, m in enumerate(self.good):
            kp_ref[i, 0] = kpts["ref"][m[0].queryIdx].pt[0]
            kp_ref[i, 1] = kpts["ref"][m[0].queryIdx].pt[1]
            kp_cur[i, 0] = kpts["cur"][m[0].trainIdx].pt[0]
            kp_cur[i, 1] = kpts["cur"][m[0].trainIdx].pt[1]
        return kp_ref, kp_cur

    def get_matched_keypoints(self, kpts, desc):
        self.match(desc)
        kp_ref, kp_cur = self.get_good_keypoints(kpts)
        return kp_ref, kp_cur


if __name__ == "__main__":
    from DataLoader.KITTILoader import KITTILoader
    from Detectors.HandcraftDetector import HandcraftDetector

    loader = KITTILoader()
    detector = HandcraftDetector("ORB")
    matcher = FrameByFrameMatcher("KNN")

    kpts = {}
    desc = {}
    imgs = {}
    for i, img in enumerate(loader):
        imgs["cur"] = img
        kpts["cur"], desc["cur"] = detector.detect(img)
        if i > 1:
            good = matcher.match(desc)
            img = matcher.draw_matches(imgs, kpts)
            cv2.imshow("track", img)
            if cv2.waitKey() == 27:
                break

        kpts["ref"], desc["ref"], imgs["ref"] = kpts["cur"], desc["cur"], imgs["cur"]
