# based on: https://github.com/syinari0123/SuperPoint-VO

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_log(log_file):
    img_ids = []
    est_points = []
    gt_points = []
    with open(log_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            tmp_data = line.split()

            img_ids.append(int(tmp_data[0]))
            est_points.append([float(x) for x in tmp_data[1:4]])
            gt_points.append([float(x) for x in tmp_data[4:7]])
    return np.array(img_ids), np.array(est_points), np.array(gt_points)


def main():
    files = glob.glob('results/*.txt')
    configs = [f.split('/')[-1].split('.')[0] for f in files]
    logs = [read_log(f) for f in files]

    # log output
    log_fopen = open("results/relative_errors", mode='w')
    print(configs, file=log_fopen)

    # plot
    figure = plt.figure()
    for log, config_name in zip(logs, configs):
        id, est_xyz, gt_xyz = log[0], log[1], log[2]

        errors = np.linalg.norm((est_xyz - gt_xyz), axis=1)
        avg_error = [np.mean(errors[:i]) for i in range(len(errors))]

        relative_errors = np.zeros((est_xyz.shape[0] - 1))
        for i in range(est_xyz.shape[0]):
            if i > 0:
                relative_est_xyz = est_xyz[i, :] - est_xyz[i - 1, :]
                relative_gt_xyz = gt_xyz[i, :] - gt_xyz[i - 1, :]
                relative_errors[i - 1] = np.linalg.norm((relative_est_xyz - relative_gt_xyz))

        plt.subplot(2, 1, 1)
        plt.plot(id, avg_error, label=config_name)

        plt.subplot(2, 1, 2)
        plt.plot(id[1:], relative_errors, label=config_name)

        print(relative_errors.mean(), end=' ', file=log_fopen)

    plt.subplot(2, 1, 1)
    plt.xlabel("FrameIndex")
    plt.ylabel("Avg Distance Error [m]")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel("FrameIndex")
    plt.ylabel("Relative Distance Error [m]")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
