import numpy as np
import cv2

from DataLoader.KITTILoader import KITTILoader
from Detectors.HandcraftDetector import HandcraftDetector
from Matchers.FrameByFrameMatcher import FrameByFrameMatcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer


def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for p in vo.kptdescs["cur"]["keypoints"]:
        # cv2.drawMarker(img, (int(p.pt[0]), int(p.pt[1])), (0, 255, 0), cv2.MARKER_CROSS)
        cv2.circle(img, (int(p.pt[0]), int(p.pt[1])), 3, (0, 255, 0))
    return img


class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        pass

    def update(self, est_xyz, gt_xyz):
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]

        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)

        error = np.linalg.norm(est - gt)

        self.errors.append(error)

        avg_error = np.mean(np.array(self.errors))

        # === drawer ==================================
        # each point
        draw_x, draw_y = int(x) + 290, int(z) + 90
        true_x, true_y = int(gt_x) + 290, int(gt_z) + 90

        # draw trajectory
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(self.traj, (10, 20), (600, 80), (0, 0, 0), -1)

        # draw text
        text = "[AvgError] %2.4fm" % (avg_error)
        cv2.putText(self.traj, text, (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        return self.traj


def run():
    kitti_config = {
        # "root_path": "./test_imgs",
        "root_path": "/mnt/data/datasets/public/KITTI/KITTI/odometry",
        "sequence": "00",
        "start": 0
    }

    loader = KITTILoader(kitti_config)
    detector = HandcraftDetector("SIFT")
    matcher = FrameByFrameMatcher("FLANN")
    absscale = AbosluteScaleComputer()
    traj_plotter = TrajPlotter()

    vo = VisualOdometry(detector, matcher, loader.cam)
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()
        R, t = vo.update(img, absscale.update(gt_pose))

        img1 = keypoints_plot(img, vo)
        img2 = traj_plotter.update(t, gt_pose[:, 3])

        cv2.imshow("keypoints", img1)
        cv2.imshow("trajectory", img2)
        if cv2.waitKey(10) == 27:
            break


if __name__ == "__main__":
    run()
