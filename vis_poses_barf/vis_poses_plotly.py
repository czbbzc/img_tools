import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation
import plotly.express as px

def visualize_poses(poses, size=0.05, bound=1, points=None):
    
    if poses.shape[-1] == 7:
        poses_trans = []
        for i in range(poses.shape[0]):
            trans = poses[i, :3]
            quat = poses[i, 3:]
            rot = Rotation.from_quat(quat).as_matrix()
            matrix_pose = np.eye(4, dtype=np.float32)
            matrix_pose[:3, :3] = rot
            matrix_pose[:3, 3] = trans
            
            trans_pose_temp = np.linalg.inv(matrix_pose)
            poses_trans.append(trans_pose_temp)
        poses_trans = np.array(poses_trans)        

    else:
        poses_trans = poses.reshape(-1, 4, 4)
    
    poses = poses_trans
    
    # poses: [B, 4, 4]
    
    pos_min = np.min(poses[:,:,3])-0.5
    pos_max = np.max(poses[:,:,3])+0.5
    
    # size = size*pos_max

    data = []
    # set up plots
    centers = []
    
    T1 = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
    
    T2 = np.array([[1,0,0,0],
                    [0,0,1,0],
                    [0,1,0,0],
                    [0,0,0,1]])

    for pose in poses:
        # print(pose)
        pose =  pose  @ T1
        # print(pose)
        
        # pose[0, 2] = -pose[0, 2]
        # pose[1, 2] = -pose[1, 2]
        # pose[2, 0] = -pose[2, 0]
        # pose[2, 1] = -pose[2, 1]
        # pose[2, 3] = -pose[2, 3]
        
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        
        # original
        # a = pos + size * pose[:3, 0] + size * pose[:3, 1] + 2*size * pose[:3, 2]
        # b = pos - size * pose[:3, 0] + size * pose[:3, 1] + 2*size * pose[:3, 2]
        # c = pos - size * pose[:3, 0] - size * pose[:3, 1] + 2*size * pose[:3, 2]
        # d = pos + size * pose[:3, 0] - size * pose[:3, 1] + 2*size * pose[:3, 2]
        
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - 2*size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - 2*size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - 2*size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - 2*size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3
        
        poses_all = np.array([pos, a, b, pos, c, d, pos, a, d, c, b])
        
        poses_1 = np.array([pos, a])
        
        four = np.array([a, b, c, d])
        
        for i in range(4):
            t1 = i%4
            t2 = (i+1)%4
            
            surface = np.array([pos, four[t1], four[t2]])
            
            data.append(dict(
                type="mesh3d",
                x=surface[:,0],
                y=surface[:,1],
                z=surface[:,2],
                flatshading=True,
                color='red',
                opacity=0.1,
            ))
            
        
        # face = np.array([pos, a, d, b, c])
        
        data.append(dict(
            type="scatter3d",
            x=poses_all[:,0],
            y=poses_all[:,1],
            z=poses_all[:,2],
            mode="lines",
            marker=dict(color='red',size=1.5),
        ))
        
        data.append(dict(
            type="scatter3d",
            x=poses_1[:,0],
            y=poses_1[:,1],
            z=poses_1[:,2],
            mode="lines",
            marker=dict(color='green',size=2),
        ))
        
        centers.append(pos)

        
    # fig = go.Figure(dict(
    #     data=data,
    #     win="poses",
    #     layout=dict(
    #         autosize=True,
    #         margin=dict(l=30,r=30,b=30,t=30,),
    #         showlegend=False,
    #         yaxis=dict(
    #             scaleanchor="x",
    #             scaleratio=1,
    #         )
    #     )
    # ))
    
    centers = np.array(centers)
    
    data.append(dict(
        type="scatter3d",
        x=centers[:,0],
        y=centers[:,1],
        z=centers[:,2],
        mode="markers",
        marker=dict(color='red',size=3),
    ))
        
    fig = go.Figure(data=data)
    
    # fig.update_xaxes(showgrid=True, zeroline=True , range=[pos_min,pos_max])
    # fig.update_yaxes(showgrid=True, zeroline=True , range=[pos_min,pos_max])
    # fig.update_zaxes(showgrid=True, zeroline=True , range=[pos_min,pos_max])
    # fig.update_zaxes(showgrid=True, zeroline=True)

    # 坐标轴设置
    fig.update_layout(
        scene = dict(

                        xaxis = dict(showgrid=True, zeroline=True , range=[pos_min,pos_max],),
                        yaxis = dict(showgrid=True, zeroline=True , range=[pos_min,pos_max],), # 标记4个标签值
                        zaxis = dict(showgrid=True, zeroline=True , range=[pos_min,pos_max],),            

                        # xaxis = dict(nticks=1, range=[pos_min,pos_max],),
                        # yaxis = dict(nticks=1, range=[pos_min,pos_max],), # 标记4个标签值
                        # zaxis = dict(nticks=1, range=[pos_min,pos_max],),
        ),
        # width=1000,
        scene_aspectmode='cube',
        margin=dict(r=0, l=0, b=0, t=0)
        )
    
    fig.show()
    

if __name__ == '__main__':
    
    # poses_path = './poses/cabin_mat.txt'
    poses_path = './poses/mao_mat.txt'

    poses = np.loadtxt(poses_path, dtype=np.float32)

    visualize_poses(poses[::2])