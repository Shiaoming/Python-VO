import os

import numpy as np
import cv2
import argparse
import yaml
import logging

from utils.tools import plot_keypoints

from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer

import open3d as o3d
from multiprocessing import Process, Queue


def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"])


class TrajPlotterO3D(object):
    def __init__(self):
        # initial
        self.errors = []

        # multiprocess
        self.queue = Queue(1)
        self.process = Process(target=self.loop)
        self.running = True
        self.process.start()

    def loop(self):
        from _queue import Empty
        # create o3d visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window('Trajectory', width=1600, height=900, top=90, left=160)
        # add coordinate system
        coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        vis.add_geometry(coordinate_system)
        # set render option
        opt = vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])
        opt.line_width = 3

        last_gt = np.array([0, 0, 0])
        last_est = np.array([0, 0, 0])
        while self.running:
            try:
                data_ = self.queue.get(block=False)

                est_line_frag = self.get_line_fragament(data_['est_xyz'], last_est, [1, 0, 0])
                gt_line_frag = self.get_line_fragament(data_['gt_xyz'], last_gt, [0, 1, 0])
                last_est = data_['est_xyz']
                last_gt = data_['gt_xyz']

                # view_control = vis.get_view_control()

                vis.add_geometry(est_line_frag, reset_bounding_box=False)
                vis.add_geometry(gt_line_frag, reset_bounding_box=False)

                vis.reset_view_point(True)
            except Empty:
                pass

            # show
            vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()

    def close(self):
        self.running = False
        self.process.join()

    def get_line_fragament(self, cur_xyz, las_xyz, line_color):
        points = [las_xyz, cur_xyz]
        lines = [[0, 1]]
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                        lines=o3d.utility.Vector2iVector(lines), )
        colors = [line_color for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def update(self, est_xyz, gt_xyz):
        self.queue.put({'est_xyz': est_xyz,
                        'gt_xyz': gt_xyz})


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


def run(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # create dataloader
    loader = create_dataloader(config["dataset"])
    # create detector
    detector = create_detector(config["detector"])
    # create matcher
    matcher = create_matcher(config["matcher"])

    absscale = AbosluteScaleComputer()
    # traj_plotter = TrajPlotter()
    traj_plotter = TrajPlotterO3D()

    # log
    fname = args.config.split('/')[-1].split('.')[0]
    log_fopen = open("results/" + fname + ".txt", mode='a')

    vo = VisualOdometry(detector, matcher, loader.cam)
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()
        R, t = vo.update(img, absscale.update(gt_pose))

        # === log writer ==============================
        print(i, t[0, 0], t[1, 0], t[2, 0], gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3], file=log_fopen)

        # === drawer ==================================
        img1 = keypoints_plot(img, vo)
        traj_plotter.update(t, gt_pose[:, 3])

        cv2.imshow("keypoints", img1)
        if cv2.waitKey() == 27:
            break

    cv2.waitKey()
    traj_plotter.close()
    # cv2.imwrite("results/" + fname + '.png', img2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')
    parser.add_argument('--config', type=str, default='params/kitti_superpoint_supergluematch.yaml',
                        help='config file')
    parser.add_argument('--logging', type=str, default='INFO',
                        help='logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL')

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])

    run(args)
