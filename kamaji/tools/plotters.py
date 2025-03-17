import numpy as np
import plotly.graph_objects as go
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter():
    def __init__(self):
        pass

    # def interactive_3d_trajectories(self,
    #     trajectories: List[np.ndarray], labels: List[str], is_static: List[bool],
    #     time_values: np.ndarray, show_markers: List[bool], line_widths: List[float]
    # ):
    #     """
    #     Create an interactive 3D plot with a slider for time adjustment, supporting multiple trajectories.
    #     Each trajectory can be static or animated, plotted with or without markers, and all have fixed axis ranges.

    #     Args:
    #         trajectories (List[np.ndarray]): List of arrays, each of shape (num_points, 3),
    #                                         where columns are X, Y, and Z coordinates for each trajectory.
    #         labels (List[str]): List of labels for each trajectory (used in the legend).
    #         is_static (List[bool]): List of booleans indicating if each trajectory should be static or animated.
    #         time_values (np.ndarray): 1D array of time values (one per time point) for slider labels.
    #         show_markers (List[bool]): List of booleans indicating if each trajectory should show markers.
    #         line_widths (List[float]): List of line widths for each trajectory.

    #     Returns:
    #         go.Figure: Plotly figure object with interactive slider.
    #     """
    #     # Sphere shenangians
    #     sphere_center = (25, 0, 10)
    #     sphere_radius = 5.0

    #     if not (len(trajectories) == len(labels) == len(is_static) == len(show_markers) == len(line_widths)):
    #         raise ValueError("The lengths of trajectories, labels, is_static, show_markers, and line_widths lists must match.")

    #     max_points = max(traj.shape[0] for traj in trajectories)

    #     if len(time_values) != max_points:
    #         raise ValueError("The length of time_values array must match the maximum number of points across trajectories.")
        
    #     # Calculate global axis limits for fixed ranges
    #     all_points = np.vstack(trajectories)
    #     x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    #     y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    #     z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    #     # Expand the ranges to account for the sphere's position and radius
    #     x_min = min(x_min, sphere_center[0] - sphere_radius)
    #     x_max = max(x_max, sphere_center[0] + sphere_radius)
    #     y_min = min(y_min, sphere_center[1] - sphere_radius)
    #     y_max = max(y_max, sphere_center[1] + sphere_radius)
    #     z_min = min(z_min, sphere_center[2] - sphere_radius)
    #     z_max = max(z_max, sphere_center[2] + sphere_radius)

    #     # Compute the maximum range and adjust all axes to have the same length
    #     max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    #     mid_x = (x_min + x_max) / 2
    #     mid_y = (y_min + y_max) / 2
    #     mid_z = (z_min + z_max) / 2

    #     x_min = mid_x - max_range / 2
    #     x_max = mid_x + max_range / 2
    #     y_min = mid_y - max_range / 2
    #     y_max = mid_y + max_range / 2
    #     z_min = mid_z - max_range / 2
    #     z_max = mid_z + max_range / 2

    #     fig = go.Figure()

    #     sphere = self.add_sphere_mesh(fig, center=sphere_center, radius=sphere_radius)
    #     fig.add_trace(sphere)

    #     # Initialize each trajectory in the base figure
    #     for i, (data, label, static, marker, line_width) in enumerate(zip(trajectories, labels, is_static, show_markers, line_widths)):
    #         x, y, z = data[:, 0], data[:, 1], data[:, 2]

    #         # Reduce number of points if trajectory has over 500 points, scale to match time_values
    #         if len(x) > 500:
    #             indices = np.linspace(0, len(x) - 1, 500, dtype=int)
    #             x, y, z = x[indices], y[indices], z[indices]
    #             time_resampled = time_values[indices]
    #         else:
    #             time_resampled = time_values

    #         initial_x, initial_y, initial_z = (x, y, z) if static else ([x[0]], [y[0]], [z[0]])

    #         fig.add_trace(go.Scatter3d(
    #             x=initial_x, y=initial_y, z=initial_z,
    #             mode="lines+markers" if marker else "lines",
    #             name=label,
    #             marker=dict(size=4),
    #             line=dict(width=line_width),  # Use specified line width
    #             showlegend=True
    #         ))

    #     # Add frames for animation
    #     frames = []
    #     for k in range(1, 500 if max_points > 500 else max_points):
    #         frame_data = [sphere]
    #         for i, (data, static, marker, line_width) in enumerate(zip(trajectories, is_static, show_markers, line_widths)):
    #             x, y, z = data[:, 0], data[:, 1], data[:, 2]

    #             if len(x) > 500:
    #                 indices = np.linspace(0, len(x) - 1, 500, dtype=int)
    #                 x, y, z = x[indices], y[indices], z[indices]

    #             if static:
    #                 frame_data.append(go.Scatter3d(
    #                     x=x, y=y, z=z,
    #                     mode="lines+markers" if marker else "lines",
    #                     marker=dict(size=4),
    #                     line=dict(width=line_width)
    #                 ))
    #             elif k < len(x):
    #                 frame_data.append(go.Scatter3d(
    #                     x=x[:k+1], y=y[:k+1], z=z[:k+1],
    #                     mode="lines+markers" if marker else "lines",
    #                     marker=dict(size=4),
    #                     line=dict(width=line_width)
    #                 ))

    #         frames.append(go.Frame(data=frame_data, name=str(k)))
        
    #     fig.frames = frames
       
    #     # Subsample time_values down to 500 points for slider display
    #     time_resampled_slider = np.linspace(0, len(time_values) - 1, 500, dtype=int)
    #     time_values_slider = time_values[time_resampled_slider]

    #     # Slider with labels based on the reduced set of time values, rounded to 2 decimal points
    #     steps = [
    #         dict(
    #             method="animate", 
    #             args=[[str(k)], dict(mode="immediate", frame=dict(duration=50, redraw=True), fromcurrent=True)],
    #             label=f"{time_values_slider[k]:.2f}"  # Display time values from the full time range
    #         ) for k in range(500)
    #     ]
        
    #     # Ensure the last step corresponds to the final time value in time_values
    #     steps.append(
    #         dict(
    #             method="animate", 
    #             args=[[str(max_points - 1)], dict(mode="immediate", frame=dict(duration=50, redraw=True), fromcurrent=True)],
    #             label=f"{time_values[-1]:.2f}"  # Ensure the final time value is displayed
    #         )
    #     )

    #     # Define the slider
    #     sliders = [dict(
    #         active=0,
    #         currentvalue=dict(prefix="Time: ", font=dict(size=20)),
    #         pad=dict(b=10),
    #         len=0.9,
    #         x=0.1,
    #         y=0,
    #         steps=steps
    #     )]

    #     fig.update_layout(
    #         title="Interactive 3D Multi-Trajectory over Time",
    #         scene=dict(
    #             xaxis=dict(title="X", range=[x_min, x_max]),
    #             yaxis=dict(title="Y", range=[y_min, y_max]),
    #             zaxis=dict(title="Z", range=[z_min, z_max]),
    #             aspectmode='cube'
    #         ),
    #         updatemenus=[dict(
    #             type="buttons",
    #             showactive=False,
    #             buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
    #                     dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])]
    #         )],
    #         sliders=sliders,
    #         showlegend=True
    #     )

    #     return fig

    def interactive_3d_trajectories(self,
        trajectories: List[np.ndarray], labels: List[str], is_static: List[bool],
        time_values: np.ndarray, show_markers: List[bool], line_widths: List[float],
        spheres: List[tuple[float, float, float, float]] = []
    ):
        """
        Create an interactive 3D plot with a slider for time adjustment, supporting multiple trajectories.
        Now includes multiple spheres defined by (x, y, z, r) tuples.

        Args:
            trajectories (List[np.ndarray]): List of arrays, each of shape (num_points, 3),
                                            where columns are X, Y, and Z coordinates for each trajectory.
            labels (List[str]): List of labels for each trajectory (used in the legend).
            is_static (List[bool]): List of booleans indicating if each trajectory should be static or animated.
            time_values (np.ndarray): 1D array of time values (one per time point) for slider labels.
            show_markers (List[bool]): List of booleans indicating if each trajectory should show markers.
            line_widths (List[float]): List of line widths for each trajectory.
            spheres (List[Tuple[float, float, float, float]]): List of tuples defining spheres (x, y, z, r).

        Returns:
            go.Figure: Plotly figure object with interactive slider.
        """
        if not (len(trajectories) == len(labels) == len(is_static) == len(show_markers) == len(line_widths)):
            raise ValueError("The lengths of trajectories, labels, is_static, show_markers, and line_widths lists must match.")

        max_points = max(traj.shape[0] for traj in trajectories)

        if len(time_values) != max_points:
            raise ValueError("The length of time_values array must match the maximum number of points across trajectories.")
        
        # Calculate global axis limits for fixed ranges
        all_points = np.vstack(trajectories)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

        # Expand the ranges to account for each sphere's position and radius
        for sphere in spheres:
            xc, yc, zc, r = sphere[0], sphere[1], sphere[2], sphere[3]
            x_min = min(x_min, xc - r)
            x_max = max(x_max, xc + r)
            y_min = min(y_min, yc - r)
            y_max = max(y_max, yc + r)
            z_min = min(z_min, zc - r)
            z_max = max(z_max, zc + r)

        # Compute the maximum range and adjust all axes to have the same length
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        mid_z = (z_min + z_max) / 2

        x_min = mid_x - max_range / 2
        x_max = mid_x + max_range / 2
        y_min = mid_y - max_range / 2
        y_max = mid_y + max_range / 2
        z_min = mid_z - max_range / 2
        z_max = mid_z + max_range / 2

        fig = go.Figure()

        sphere_meshes = []
        # Add spheres to the plot
        for sphere in spheres:
            radius = sphere[3]
            center = sphere[0:3]
            sphere = self.add_sphere_mesh(fig, center=center, radius=radius)
            sphere_meshes.append(sphere)
            fig.add_trace(sphere)

        # Initialize each trajectory in the base figure
        for i, (data, label, static, marker, line_width) in enumerate(zip(trajectories, labels, is_static, show_markers, line_widths)):
            x, y, z = data[:, 0], data[:, 1], data[:, 2]

            # Reduce number of points if trajectory has over 500 points, scale to match time_values
            if len(x) > 500:
                indices = np.linspace(0, len(x) - 1, 500, dtype=int)
                x, y, z = x[indices], y[indices], z[indices]
                time_resampled = time_values[indices]
            else:
                time_resampled = time_values

            initial_x, initial_y, initial_z = (x, y, z) if static else ([x[0]], [y[0]], [z[0]])

            fig.add_trace(go.Scatter3d(
                x=initial_x, y=initial_y, z=initial_z,
                mode="lines+markers" if marker else "lines",
                name=label,
                marker=dict(size=4),
                line=dict(width=line_width),
                showlegend=True
            ))

        # Add frames for animation
        frames = []
        for k in range(1, 500 if max_points > 500 else max_points):
            frame_data = []
            for mesh in sphere_meshes:
                frame_data.append(mesh)
            for i, (data, static, marker, line_width) in enumerate(zip(trajectories, is_static, show_markers, line_widths)):
                x, y, z = data[:, 0], data[:, 1], data[:, 2]

                if len(x) > 500:
                    indices = np.linspace(0, len(x) - 1, 500, dtype=int)
                    x, y, z = x[indices], y[indices], z[indices]

                if static:
                    frame_data.append(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode="lines+markers" if marker else "lines",
                        marker=dict(size=4),
                        line=dict(width=line_width)
                    ))
                elif k < len(x):
                    frame_data.append(go.Scatter3d(
                        x=x[:k+1], y=y[:k+1], z=z[:k+1],
                        mode="lines+markers" if marker else "lines",
                        marker=dict(size=4),
                        line=dict(width=line_width)
                    ))

            frames.append(go.Frame(data=frame_data, name=str(k)))
        
        fig.frames = frames

        # Subsample time_values down to 500 points for slider display
        time_resampled_slider = np.linspace(0, len(time_values) - 1, 500, dtype=int)
        time_values_slider = time_values[time_resampled_slider]

        # Slider with labels based on the reduced set of time values, rounded to 2 decimal points
        steps = [
            dict(
                method="animate", 
                args=[[str(k)], dict(mode="immediate", frame=dict(duration=5, redraw=True), fromcurrent=True)],
                label=f"{time_values_slider[k]:.2f}"
            ) for k in range(500)
        ]

        steps.append(
            dict(
                method="animate", 
                args=[[str(max_points - 1)], dict(mode="immediate", frame=dict(duration=5, redraw=True), fromcurrent=True)],
                label=f"{time_values[-1]:.2f}"
            )
        )

        sliders = [dict(
            active=0,
            currentvalue=dict(prefix="Time: ", font=dict(size=20)),
            pad=dict(b=10),
            len=0.9,
            x=0.1,
            y=0,
            steps=steps
        )]

        fig.update_layout(
            title="Interactive 3D Multi-Trajectory over Time",
            scene=dict(
                xaxis=dict(title="X", range=[x_min, x_max]),
                yaxis=dict(title="Y", range=[y_min, y_max]),
                zaxis=dict(title="Z", range=[z_min, z_max]),
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=5, redraw=True), fromcurrent=True)]),
                        dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])]
            )],
            sliders=sliders,
            showlegend=True
        )

        return fig

    
    def add_sphere_mesh(self, fig, center, radius, sphere_resolution=30):
        """
        Add a complete, undistorted 3D sphere to the given Plotly figure.

        Args:
            fig (go.Figure): The figure to add the sphere to.
            center (tuple): The (X, Y, Z) coordinates of the sphere's center.
            radius (float): Radius of the sphere.
            sphere_resolution (int): Number of divisions for longitude and latitude (higher gives finer resolution).
        """
        theta_vals = np.linspace(0, np.pi, sphere_resolution)  # Polar angle
        phi_vals = np.linspace(0, 2 * np.pi, sphere_resolution)  # Azimuthal angle
        
        # Compute the spherical coordinates
        x_sphere = radius * np.outer(np.sin(theta_vals), np.cos(phi_vals)) + center[0]
        y_sphere = radius * np.outer(np.sin(theta_vals), np.sin(phi_vals)) + center[1]
        z_sphere = radius * np.outer(np.cos(theta_vals), np.ones_like(phi_vals)) + center[2]

        # Flatten arrays for plotting in Mesh3d
        x_flat = x_sphere.ravel()
        y_flat = y_sphere.ravel()
        z_flat = z_sphere.ravel()

        # Create face indices for triangles
        i, j, k = [], [], []
        for lat in range(sphere_resolution - 1):
            for lon in range(sphere_resolution - 1):
                p1 = lat * sphere_resolution + lon
                p2 = p1 + sphere_resolution
                p3 = p1 + 1
                p4 = p2 + 1

                # Create two triangles for each square patch
                i.extend([p1, p2, p2])
                j.extend([p2, p3, p3])
                k.extend([p3, p1, p4])

        # Add the sphere as a Mesh3d trace
        return go.Mesh3d(
            x=x_flat, y=y_flat, z=z_flat,
            i=i, j=j, k=k,
            opacity=0.5,  # Adjust as desired
            color='lightblue',  # Adjust color as desired
            name="Sphere",
            showlegend=True
        )

    def plot_3d_trajectories(self, trajectories, labels=None, linewidths=None, markers=None, sphere_centers=None, sphere_radii=None):
        """
        Creates a 3D plot for multiple trajectories, with options for labels, line widths, and markers, 
        as well as plotting spheres at specified centers with specified radii.

        Args:
            trajectories (list of np.ndarray): List of numpy arrays, each of shape (num_points, 3)
                                                representing X, Y, Z coordinates for different trajectories.
            labels (list of str, optional): List of labels for the trajectories. Default is None.
            linewidths (list of float, optional): List of linewidths for the trajectories. Default is None.
            markers (list of str, optional): List of marker types for the trajectories. Default is None.
            sphere_centers (list of tuple, optional): List of (X, Y, Z) coordinates for the centers of spheres. Default is None.
            sphere_radii (list of float, optional): List of radii for the spheres. Default is None.

        Returns:
            None: Displays the 3D plot with the given trajectories and spheres.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Check if the length of optional parameters match the number of trajectories
        num_trajectories = len(trajectories)
        
        if labels and len(labels) != num_trajectories:
            raise ValueError("The number of labels must match the number of trajectories.")
        
        if linewidths and len(linewidths) != num_trajectories:
            raise ValueError("The number of linewidths must match the number of trajectories.")
        
        if markers and len(markers) != num_trajectories:
            raise ValueError("The number of markers must match the number of trajectories.")

        # Iterate through the list of trajectories and plot each one with the given properties
        for i, trajectory in enumerate(trajectories):
            X, Y, Z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
            
            label = labels[i] if labels else None
            linewidth = linewidths[i] if linewidths else 1  # Default linewidth is 1
            marker = markers[i] if markers else 'o'  # Default marker is 'o'

            ax.plot(X, Y, Z, label=label, linewidth=linewidth, marker=marker)

        # Plot spheres if provided
        if sphere_centers and sphere_radii:
            if len(sphere_centers) != len(sphere_radii):
                raise ValueError("The number of sphere centers must match the number of sphere radii.")
            
            for center, radius in zip(sphere_centers, sphere_radii):
                # Create a meshgrid for the sphere surface
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

                # Plot the sphere
                ax.plot_surface(x, y, z, color='r', alpha=0.5)

        # Set axis labels and title
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # ax.set_title('3D Trajectories with Spheres')
        ax.set_aspect('equal')

        # Add legend if labels are provided
        if labels:
            ax.legend()

        plt.show()
