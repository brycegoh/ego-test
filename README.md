# Task

Given an egocentric dataset, make it compatible to use to train a bimanual mobile manipulation robot with parallel grippers.

# Challenges

1. hand to gripper pose translation
2. hand is present in camera frame

# Goal

Given a video frame of the top, left wrist and right wrist, recreate it such that the video contains the robot arm positioned like the hand. When grapping things, it also chooses a sensible orientation.

# Existing Literature

Dex-umi that inpained the human hand with robotic hand
https://dex-umi.github.io/static/pdfs/DexUMI:%20Using%20Human%20Hand%20as%20the%20Universal%20Manipulation%20Interface%20for%20Dexterous%20Manipulation.pdf