#### pose evaluation
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
import os
from utils import *
import glob

result_path = "/dev_ws/results/replica_rgbd_0/office0/" 
gt_path = "/dev_ws/data/Replica/office0/" 
benchmark_path ="/dev_ws/benchmark"
show_plot = False
correct_scale = False


def loadReplica(path):
    color_paths = sorted(glob.glob(os.path.join(path, "results/frame*.jpg")))
    #print(path, color_paths)
    tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp


#load gt
if "replica" in gt_path.lower():
    print("loading replica")
    gt_color_paths, gt_tstamp = loadReplica(gt_path)


pose_path = os.path.join(result_path, "CameraTrajectory_TUM.txt")
poses, tstamp = loadPose(pose_path)

# load estimated poses
traj_est = file_interface.read_tum_trajectory_file(pose_path)
# load gt pose
if "kitti" in gt_path.lower():
    def loadKITTIPose(gt_path):
        scene = gt_path.split("/")[-1]
        gt_file = gt_path.replace(scene, 'poses/{}.txt'.format(scene))
        pose_quat = []
        with open(gt_file, "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].split()
                #print(line)
                c2w = np.array(list(map(float, line))).reshape(3, 4)
                #print(c2w)
                quat = np.zeros(7)
                quat[:3] = c2w[:3, 3]
                quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                pose_quat.append(quat)
        pose_quat = np.array(pose_quat)

        return pose_quat
        
    pose_quat = loadKITTIPose(gt_path)
    traj_ref = PoseTrajectory3D(positions_xyz=pose_quat[:,:3],         
                                    orientations_quat_wxyz=pose_quat[:,3:],         
                                    timestamps=np.array(gt_tstamp))

    #scene = args.gt_path.split("/")[-1]
    #gt_file = args.gt_path.replace(scene, 'poses/{}.txt'.format(scene))
    #print(gt_file)
    #traj_ref = file_interface.read_kitti_poses_file(gt_file)
elif "replica" in gt_path.lower():
    gt_file = os.path.join(gt_path, 'pose_TUM.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    #if not os.path.isfile(gt_file):
elif "euroc" in gt_path.lower():
    gt_file = os.path.join(gt_path, 'mav0/state_groundtruth_estimate0/data.csv')
    traj_ref = file_interface.read_euroc_csv_trajectory(gt_file)
else:
    gt_file = os.path.join(gt_path, 'groundtruth.txt')   
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.1)
result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
    pose_relation=PoseRelation.translation_part, align=False, correct_scale=correct_scale)
result_rotation_part = main_ape.ape(traj_ref, traj_est, est_name='rot', pose_relation=PoseRelation.rotation_part, 
                                    align=False, correct_scale=correct_scale)

out_path=os.path.join(benchmark_path, "metrics_traj.txt")
with open(out_path, 'w') as fp:
    fp.write(result.pretty_str())
    fp.write(result_rotation_part.pretty_str())
print(result)

if show_plot:
    from evo.tools import plot
    from evo.tools.plot import PlotMode
    import matplotlib.pyplot as plt
    import copy
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True)
    fig = plt.figure()
    traj_by_label = {
        "estimate (not aligned)": traj_est,
        "estimate (aligned)": traj_est_aligned,
        "reference": traj_ref
    }
    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    plt.show()
