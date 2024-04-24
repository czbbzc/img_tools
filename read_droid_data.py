import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import os


def read_droid_data(read_path, save_path=None):
    
    imgs = np.load(os.path.join(read_path, 'images.npy'))
    poses_quat = np.load(os.path.join(read_path, 'poses.npy'))    # 四元数格式
    intrinsics = np.load(os.path.join(read_path, 'intrinsics.npy'))
    
    if len(imgs.shape) == 4:
        imgs = imgs.transpose(0, 2, 3, 1)
    else:
        imgs = imgs.transpose(1, 2, 0)
        
    # 四元数 -> 转移矩阵
    poses_matrix = []
    for i in range(poses_quat.shape[0]):
        trans = poses_quat[i, :3]
        quat = poses_quat[i, 3:]
        rot = Rotation.from_quat(quat).as_matrix()
        matrix_pose = np.eye(4, dtype=np.float32)
        matrix_pose[:3, :3] = rot
        matrix_pose[:3, 3] = trans
        
        trans_pose_temp = np.linalg.inv(matrix_pose)

        poses_matrix.append(trans_pose_temp)
    poses_matrix = np.array(poses_matrix, dtype=np.float32)
    
    intrinsics = intrinsics*8
    
    # save
    if save_path != None:
        imgs_path = os.path.join(save_path, 'images')
        poses_quat_path = os.path.join(save_path, 'poses_quat.txt')
        poses_matrix_path = os.path.join(save_path, 'poses_matrix.txt')
        intrinsics_path = os.path.join(save_path, 'cameras.txt')
        os.makedirs(imgs_path, exist_ok=True)
        
        with open(intrinsics_path, 'w') as f:
            np.savetxt(f, intrinsics[0].reshape(1, -1))
        
        for i in range(imgs.shape[0]):
            
            # save image
            img_path = os.path.join(imgs_path, 'rgb_{:0>5d}.png'.format(i))
            cv2.imwrite(img_path, imgs[i])
            
            # save poses
            if i == 0:
                # quat
                with open(poses_quat_path, 'w') as f:
                    np.savetxt(f, poses_quat[i].reshape(1, -1))
                # matrix
                with open(poses_matrix_path, 'w') as f:
                    np.savetxt(f, poses_matrix[i].reshape(1, -1))
            else:
                # quat
                with open(poses_quat_path, 'a') as f:
                    np.savetxt(f, poses_quat[i].reshape(1, -1))
                # matrix
                with open(poses_matrix_path, 'a') as f:
                    np.savetxt(f, poses_matrix[i].reshape(1, -1))
            
    
            print(img_path)
    
    print('over!')
    
    
if __name__ == '__main__':
    read_path = 'busters/art'
    save_path = 'busters/art_save'
    read_droid_data(read_path, save_path)