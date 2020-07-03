import numpy as np
import cv2
import logging
from utils.tools import dict_update, plot_matches


class FrameByFrameMatcher(object):
    default_config = {
        "type": "FLANN",
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

    def __init__(self, config=None):
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        logging.info("Frame by frame matcher config: ")
        logging.info(self.config)

        if self.config["type"] == "KNN":
            logging.info("creating brutal force matcher...")
            if self.config["KNN"]["HAMMING"]:
                logging.info("brutal force with hamming norm.")
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.matcher = cv2.BFMatcher()
        elif self.config["type"] == "FLANN":
            logging.info("creating FLANN matcher...")
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=self.config["FLANN"]["kdTrees"])
            search_params = dict(checks=self.config["FLANN"]["searchChecks"])  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher type: {self.matcher_type}")

    def match(self, kptdescs):
        self.good = []
        if self.config["type"] == "KNN" and self.config["KNN"]["HAMMING"]:
            logging.debug("KNN keypoints matching...")
            matches = self.matcher.match(kptdescs["ref"]["descriptors"], kptdescs["cur"]["descriptors"])
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            # self.good = matches[:self.config["KNN"]["first_N"]]
            for i in range(self.config["KNN"]["first_N"]):
                self.good.append([matches[i]])
        else:
            logging.debug("FLANN keypoints matching...")
            matches = self.matcher.knnMatch(kptdescs["ref"]["descriptors"], kptdescs["cur"]["descriptors"], k=2)
            # Apply ratio test
            for m, n in matches:
                if m.distance < self.config["distance_ratio"] * n.distance:
                    self.good.append([m])
        return self.good

    def get_good_keypoints(self, kptdescs):
        logging.debug("getting matched keypoints...")
        kp_ref = np.zeros([len(self.good), 2])
        kp_cur = np.zeros([len(self.good), 2])
        match_score = np.zeros([len(self.good)])
        for i, m in enumerate(self.good):
            kp_ref[i, :] = kptdescs["ref"]["keypoints"][m[0].queryIdx]
            kp_cur[i, :] = kptdescs["cur"]["keypoints"][m[0].trainIdx]
            match_score[i] = m[0].distance

        ret_dict = {
            "ref_keypoints": kp_ref,
            "cur_keypoints": kp_cur,
            "match_score": match_score
        }
        return ret_dict

    def __call__(self, kptdescs):
        self.match(kptdescs)
        return self.get_good_keypoints(kptdescs)


if __name__ == "__main__":
    from DataLoader.KITTILoader import KITTILoader
    from Detectors.HandcraftDetector import HandcraftDetector

    loader = KITTILoader()
    detector = HandcraftDetector({"type": "SIFT"})
    matcher = FrameByFrameMatcher({"type": "FLANN"})

    kptdescs = {}
    imgs = {}
    for i, img in enumerate(loader):
        imgs["cur"] = img
        kptdescs["cur"] = detector(img)
        if i > 1:
            matches = matcher(kptdescs)
            img = plot_matches(imgs['ref'], imgs['cur'],
                               matches['ref_keypoints'], matches['cur_keypoints'],
                               matches['match_score'], layout='ud')
            cv2.imshow("track", img)
            if cv2.waitKey() == 27:
                break

        kptdescs["ref"], imgs["ref"] = kptdescs["cur"], imgs["cur"]
