import cv2
import numpy as np
from utils.tools import dict_update, plot_keypoints


class HandcraftDetector(object):
    default_config = {
        "ORB": {
            "nfeatures": 1000,
            "scaleFactor": 1.2,
            "nLevels": 8,
            "edgeThreshold": 31,
            "firstLevel": 0,
            "WTA_K": 2,
            "patchSize": 31,
            "fastThreshold": 20
        },
        "SIFT": {
            "nfeatures": 1000,
            "nOctaveLayers": 3,
            "contrastThreshold": 0.04,
            "edgeThreshold": 10,
            "sigma": 1.6
        }
    }

    def __init__(self, feature_type="ORB", config=None):
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.feature_type = feature_type

        if self.feature_type == "ORB":
            self.det = cv2.ORB_create(nfeatures=self.config["ORB"]["nfeatures"],
                                      scaleFactor=self.config["ORB"]["scaleFactor"],
                                      nlevels=self.config["ORB"]["nLevels"],
                                      edgeThreshold=self.config["ORB"]["edgeThreshold"],
                                      firstLevel=self.config["ORB"]["firstLevel"],
                                      WTA_K=self.config["ORB"]["WTA_K"],
                                      patchSize=self.config["ORB"]["patchSize"],
                                      fastThreshold=self.config["ORB"]["fastThreshold"])
        elif self.feature_type == "SIFT":
            self.det = cv2.xfeatures2d.SIFT_create(nfeatures=self.config["SIFT"]["nfeatures"],
                                                   nOctaveLayers=self.config["SIFT"]["nOctaveLayers"],
                                                   contrastThreshold=self.config["SIFT"]["contrastThreshold"],
                                                   edgeThreshold=self.config["SIFT"]["edgeThreshold"],
                                                   sigma=self.config["SIFT"]["sigma"]
                                                   )
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kpts_cv, desc = self.det.detectAndCompute(image, None)

        kpts = np.zeros((len(kpts_cv), 2))
        scores = np.zeros((len(kpts_cv)))
        for i, p in enumerate(kpts_cv):
            kpts[i, 0] = p.pt[0]
            kpts[i, 1] = p.pt[1]
            scores[i] = p.response

        return {"keypoints": kpts,
                "scores": scores,
                "descriptors": desc}


if __name__ == "__main__":
    img0 = cv2.imread("../test_imgs/sequences/00/image_0/000000.png")

    handcraft_detector = HandcraftDetector("SIFT")
    kptdesc = handcraft_detector(img0)

    img = plot_keypoints(img0, kptdesc["keypoints"], kptdesc["scores"])
    cv2.imshow("SIFT", img)
    cv2.waitKey()
