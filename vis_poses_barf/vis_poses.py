import numpy as np
import camera0
import visdom
import torch
from scipy.spatial.transform import Rotation


def get_camera_mesh(pose,depth=1):
    
    # vertices = np.array([[-0.5,-0.5,1],
    #                          [0.5,-0.5,1],
    #                          [0.5,0.5,1],
    #                          [-0.5,0.5,1],
    #                          [0,0,0]])*depth
    # faces = np.array([[0,1,2],
    #                       [0,2,3],
    #                       [0,1,4],
    #                       [1,2,4],
    #                       [2,3,4],
    #                       [3,0,4]])
    
    vertices = torch.tensor([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])
    vertices = camera0.cam2world(vertices[None],pose)
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices,faces,wireframe

def merge_wireframes(wireframe):
    wireframe_merged = [[],[],[]]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:,0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:,1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:,2]]+[None]
    return wireframe_merged
def merge_meshes(vertices,faces):
    mesh_N,vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)],dim=0)
    
    # faces_merged = np.concatenate([faces+i*vertex_N for i in range(mesh_N)],axis=0)
    # input=np.concatenate((self._A, self._B),axi=1)
    
    vertices_merged = vertices.view(-1,vertices.shape[-1])
    return vertices_merged,faces_merged

def merge_centers(centers):
    center_merged = [[],[],[]]
    for c1,c2 in zip(*centers):
        center_merged[0] += [float(c1[0]),float(c2[0]),None]
        center_merged[1] += [float(c1[1]),float(c2[1]),None]
        center_merged[2] += [float(c1[2]),float(c2[2]),None]
    return center_merged

def vis_cameras(group, name, poses=[],colors=["blue"],plot_dist=False,step=0):
    
    # colors=["blue","magenta"]
    
    vis = visdom.Visdom(server="localhost",port=9000,env=group)
    
    win_name = "{}/{}".format(group,name)
    data = []
    # set up plots
    centers = []
    for pose,color in zip(poses,colors):
        pose = pose.detach().cpu()
        vertices,faces,wireframe = get_camera_mesh(pose,depth=0.2)
        center = vertices[:,-1]
        centers.append(center)
        # camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:,0]],
            y=[float(n) for n in center[:,1]],
            z=[float(n) for n in center[:,2]],
            mode="markers",
            marker=dict(color=color,size=3),
        ))
        # colored camera mesh
        vertices_merged,faces_merged = merge_meshes(vertices,faces)
        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:,0]],
            y=[float(n) for n in vertices_merged[:,1]],
            z=[float(n) for n in vertices_merged[:,2]],
            i=[int(n) for n in faces_merged[:,0]],
            j=[int(n) for n in faces_merged[:,1]],
            k=[int(n) for n in faces_merged[:,2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))
        # camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        data.append(dict(
            type="scatter3d",
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode="lines",
            line=dict(color=color,),
            opacity=0.3,
        ))
    if plot_dist:
        # distance between two poses (camera centers)
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red",width=4,),
        ))
        if len(centers)==4:
            center_merged = merge_centers(centers[2:4])
            data.append(dict(
                type="scatter3d",
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode="lines",
                line=dict(color="red",width=4,),
            ))
    # send data to visdom
    vis._send(dict(
        data=data,
        win="poses",
        eid=win_name,
        layout=dict(
            title="({})".format(step),
            autosize=True,
            margin=dict(l=30,r=30,b=30,t=30,),
            showlegend=False,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
        opts=dict(title="{} poses ({})".format(win_name,step),),
    ))
    
if __name__ == '__main__':
    
    poses_path = '/home/perple/czbbzc/repos/nerfacto_online/vis_poses/mao_quat.txt'
    
    poses = np.loadtxt(poses_path, dtype=np.float32)
    
    if poses.shape[-1] == 16:
        poses_trans = poses.reshape(-1, 4, 4)
    else:
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
        
    
    poses_trans = torch.from_numpy(poses_trans[::1][:,:3,:])    
    
    vis_cameras('mao', '1', poses=[poses_trans])