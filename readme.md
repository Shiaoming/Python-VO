## Python-VO
A simple python implemented visual odometry.

**SIFT Features on KITTI**

![sift_keypoints](screenshots/sift_keypoints.png)

**frame by frame track with SIFT Features on KITTI**

![sift_trajectory](screenshots/sift_trajectory.png)

**TODO:**
-[x] frame by frame match vo.
-[ ] SuperPoint Feature detector.
-[ ] SuperGlue Feature matcher.


-[ ] optical flow based feature track vo.

-[ ] evaluations

## Note
to use SIFT, opencv-python are build from source with opencv-contrib support (with OPENCV_ENABLE_NONFREE=ture)