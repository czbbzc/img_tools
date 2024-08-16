import numpy as np
from vis_poses_ngp.vis_poses_plt import visualize_poses
from scipy.spatial.transform import Rotation

def quat2mat(pose):
    """ convert (t, q) to 4x4 pose matrix """
    trans = pose[:3]
    quat = pose[3:]
    rot = Rotation.from_quat(quat).as_matrix()
    matrix_pose = np.eye(4, dtype=np.float32)
    matrix_pose[:3, :3] = rot
    matrix_pose[:3, 3] = trans
    
    trans_pose = np.linalg.inv(matrix_pose)
    return trans_pose



def mat2quat(pose):
    """ convert 4x4 pose matrix to (t, q) """
    
    trans_pose = np.linalg.inv(pose)    
    trans = trans_pose[:3, 3]
    rot = trans_pose[:3, :3]
    quat = Rotation.from_matrix(rot).as_quat()
    return np.concatenate([trans, quat], axis=0)
    
    # q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    # return -np.concatenate([pose[:3, 3], -q], axis=0)

mat_data = np.loadtxt('poses/autel_bicycle_mat.txt').reshape(-1, 4, 4)
quat_data = np.loadtxt('poses/autel_bicycle_quat.txt')

visualize_poses(mat_data)

trans_data = np.zeros((quat_data.shape[0], 7))
for i in range(quat_data.shape[0]):
    trans_data[i] = mat2quat(mat_data.reshape(-1,4,4)[i])
    print(trans_data[i])
    print(quat_data[i])

# trans_data = np.zeros((mat_data.shape))
# for i in range(quat_data.shape[0]):
#     trans_data[i] = quat2mat(quat_data[i])
#     print(trans_data[i])
#     print(mat_data[i])
    
visualize_poses(trans_data)