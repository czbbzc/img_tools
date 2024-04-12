import trimesh
import numpy as np


def visualize_poses(poses, size=0.1, bound=1, points=None):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]
    # objects = []

    # if bound > 1:
    #     unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
    #     unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
    #     objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        # segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    if points is not None:
        print('[visualize points]', points.shape, points.dtype, points.min(0), points.max(0))
        colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
        colors[:, 2] = 255 # blue
        colors[:, 3] = 30 # transparent
        objects.append(trimesh.PointCloud(points, colors))

    scene = trimesh.Scene(objects)
    scene.set_camera(distance=bound, center=[0, 0, 0])
    scene.show()
    

if __name__ == '__main__':
    
    poses_path = './vis_poses_ngp/mao_mat.txt'

    poses = np.loadtxt(poses_path, dtype=np.float32).reshape(-1, 4, 4)

    poses = poses[::2]

    visualize_poses(poses)