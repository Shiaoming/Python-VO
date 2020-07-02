import cv2
from utils.tools import dict_update


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

    def detect(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kpts, desc = self.det.detectAndCompute(image, None)

        return kpts, desc


if __name__ == "__main__":
    img0 = cv2.imread("../test_imgs/sequences/00/image_0/000000.png")

    orb_detector = HandcraftDetector("SIFT")
    kpts, desc = orb_detector.detect(img0)

    img = None
    img = cv2.drawKeypoints(img0, kpts, img)
    cv2.imshow("SIFT", img)
    cv2.waitKey()
