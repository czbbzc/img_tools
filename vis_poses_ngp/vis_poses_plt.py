import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_poses(poses, size=0.1, bound=1, points=None):
    # poses: [B, 4, 4]
    
    pos_min = np.min(poses)
    pos_max = np.max(poses)
    
    #定义坐标轴
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    # ax1=plt.axes(projection='3d')

    #设置坐标轴
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    plt.xlim(pos_min,pos_max)
    plt.ylim(pos_min,pos_max)
    # ax1.zlim(pos_min,pos_max)


    for pose in poses:
        # a camera is visualized with 8 line segments.
        
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        
        poses_all  = np.array([a, b, c, d])

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        
        for i in range(4):
            ax1.plot([pos[0], poses_all[i][0]], [pos[1], poses_all[i][1]], [pos[2], poses_all[i][2]], color='r', linewidth=0.5)
            
            t1 = (i)%4
            t2 = (i+1)%4
            ax1.plot([poses_all[t1][0], poses_all[t2][0]], [poses_all[t1][1], poses_all[t2][1]], [poses_all[t1][2], poses_all[t2][2]], color='r', linewidth=0.5)
        

    # if points is not None:
    #     print('[visualize points]', points.shape, points.dtype, points.min(0), points.max(0))
    #     colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
    #     colors[:, 2] = 255 # blue
    #     colors[:, 3] = 30 # transparent
    #     objects.append(trimesh.PointCloud(points, colors))

    plt.show()
    

if __name__ == '__main__':
    
    poses_path = './vis_poses_ngp/mao_mat.txt'

    poses = np.loadtxt(poses_path, dtype=np.float32).reshape(-1, 4, 4)

    poses = poses[::4]

    visualize_poses(poses)