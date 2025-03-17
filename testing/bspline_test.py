import numpy as np
from scipy.interpolate import splprep, splev
from uam.tools import tools
import matplotlib.pyplot as plt

def fit_b_spline(data_points, num_samples=2000, smoothing=0):
    """
    Fits a B-spline curve to the given data points and returns a dense sampling of points along the curve.

    Args:
        data_points (np.ndarray): Array of shape (num_points, 3) representing X, Y, and Z coordinates.
        num_samples (int): The number of points to sample along the B-spline curve. Default is 2000.
        smoothing (float): Smoothing factor for the spline fit. Default is 0 (no smoothing).

    Returns:
        np.ndarray: Array of shape (num_samples, 3) representing sampled points on the fitted B-spline curve.
    """
    # Ensure the input data has the correct shape
    if data_points.shape[1] != 3:
        raise ValueError("data_points array must be of shape (num_points, 3)")
    
    # Prepare the spline parameters using SciPy's splprep function
    tck, u = splprep(data_points.T, s=smoothing)
    
    # Generate the parameter values for the desired number of samples
    u_dense = np.linspace(0, 1, num_samples)
    
    # Evaluate the B-spline at the dense sample points
    spline_points = splev(u_dense, tck)
    
    # Convert the spline points back to (num_samples, 3) format
    spline_points = np.array(spline_points).T
    
    return spline_points

if __name__ == "__main__":
    helix_sparse = tools.generate_3d_helix(10, 10, 5, 25)
    helix_full = tools.generate_3d_helix(2000, 10, 5, 25)

    helix_bspline = fit_b_spline(helix_sparse, 2000, smoothing=0)

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the helix
    ax.plot(helix_full[:, 0], helix_full[:, 1], helix_full[:, 2], color='k', label='Full')
    ax.plot(helix_bspline[:, 0], helix_bspline[:, 1], helix_bspline[:, 2], color='b', label='B-Spline')

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Helix')

    # Set equal scaling
    max_range = np.array([helix_full[:, 0].max() - helix_full[:, 0].min(),
                        helix_full[:, 1].max() - helix_full[:, 1].min(),
                        helix_full[:, 2].max() - helix_full[:, 2].min()]).max() / 2.0

    mid_x = (helix_full[:, 0].max() + helix_full[:, 0].min()) * 0.5
    mid_y = (helix_full[:, 1].max() + helix_full[:, 1].min()) * 0.5
    mid_z = (helix_full[:, 2].max() + helix_full[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()

    # Show the plot
    plt.show()
