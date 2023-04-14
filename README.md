# pose_estimate
This repo gives some ways to recover camera pose from the motion.
- pose_2d2d: Directly computing the essential and fundamental matrix to recover cam pose.
# How to run?
```bash
rosrun pose_estimate pose_2d2d src/pose_estimate/img/1.png src/pose_estimate/img/2.png
```