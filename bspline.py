import numpy as np
from scipy.interpolate import splprep, splev
from vis_poses_ngp.vis_poses_plt import visualize_poses

def interpolate_b_spline(data, num_samples):
    """
    Interpolates a series of data points using B-spline interpolation.

    Parameters:
    - data: List of data points, where each point is a tuple of (x, y, z, q0, q1, q2, q3)
    - num_samples: Number of samples for the interpolated trajectory

    Returns:
    - interpolated_data: List of interpolated data points
    """

    # Separate xyz coordinates and quaternions
    xyz = np.array([point[:3] for point in data])
    quaternions = np.array([point[3:] for point in data])

    # Compute cumulative distance along the trajectory
    cumulative_distance = np.zeros(len(data))
    for i in range(1, len(data)):
        cumulative_distance[i] = cumulative_distance[i-1] + np.linalg.norm(xyz[i] - xyz[i-1])

    # Normalize cumulative distance to [0, 1]
    normalized_distance = cumulative_distance / cumulative_distance[-1]

    # B-spline interpolation for xyz coordinates
    tck_xyz, _ = splprep(xyz.T, u=normalized_distance, k=len(data)-1, s=0)
    u_new = np.linspace(0, 1, num_samples)
    interpolated_xyz = splev(u_new, tck_xyz)

    # B-spline interpolation for quaternions
    tck_quat, _ = splprep(quaternions.T, u=normalized_distance, k=len(data)-1, s=0)
    interpolated_quaternions = splev(u_new, tck_quat)

    # Combine interpolated xyz and quaternions
    interpolated_data = [(interpolated_xyz[0][i], interpolated_xyz[1][i], interpolated_xyz[2][i],
                          interpolated_quaternions[0][i], interpolated_quaternions[1][i],
                          interpolated_quaternions[2][i], interpolated_quaternions[3][i])
                         for i in range(len(u_new))]

    return interpolated_data

# Example usage:
# data = [(1, 2, 3, 0.707, 0, 0.707, 0), (2, 3, 4, 0.5, 0.5, 0.5, 0.5), (3, 4, 5, 0, 0.707, 0, 0.707)]

init_poses = np.loadtxt('vis_poses_ngp/zkz_quat.txt')
fs = np.linspace(0, len(init_poses)-1, 6, dtype=np.uint8)
data = init_poses[fs]

visualize_poses(data)

num_samples = 12
interpolated_data = interpolate_b_spline(data, num_samples)

visualize_poses(np.array(interpolated_data))
