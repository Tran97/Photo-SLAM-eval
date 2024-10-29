import os
import numpy as np
from scipy.spatial.transform import Rotation

import argparse
parser = argparse.ArgumentParser(description="This is a script convert Replica camera pose file to TUM_camera_pose format")
parser.add_argument("-d", "--replica_dataset_path", type=str, required=True)
args = parser.parse_args()

def load_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        poses.append(c2w)
    return poses

def save_pose_as_kitti(path, poses):
    with open(path, "w") as f:
        for pose in poses:
            line = "{} {} {} {} {} {} {} {} {} {} {} {}\n".format(pose[0,0], pose[0,1], pose[0,2], pose[0,3], pose[1,0], pose[1,1], pose[1,2], pose[1,3], pose[2,0], pose[2,1], pose[2,2], pose[2,3])
            f.write(line)

def save_pose_as_tum(path, poses):
    i = 0
    with open(path, "w") as f:
        for pose in poses:
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            line = "{} {} {} {} {} {} {} {}\n".format(i, pose[0, 3], pose[1, 3], pose[2, 3], quat[0], quat[1], quat[2], quat[3])
            f.write(line)
            i+=1

# Normalize path
dataset_path = os.path.normpath(args.replica_dataset_path)

# Loop over folders in dataset path
folders = os.listdir(args.replica_dataset_path)
for folder in folders:
    folder_path = os.path.join(dataset_path, folder.strip())  # Strip whitespace from folder names
    # only process if the item is a directory
    if os.path.isdir(folder_path):
        traj_path = os.path.join(folder_path, "traj.txt").strip()  # strip whitespace from traj.txt path   
                # Check if traj.txt exists as a file, not a directory or symlink
        if os.path.isfile(traj_path):
            poses = load_poses(traj_path)
            
            if poses:  # Ensure poses were loaded successfully
                tum_output_path = traj_path.replace("traj.txt", "GTpose_TUM_format.txt")
                save_pose_as_tum(tum_output_path, poses)
                