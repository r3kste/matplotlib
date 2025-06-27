import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyArrow
from mpl_toolkits.mplot3d import proj3d


def arrows3d(end, start=None, ax=None, label=None, color=None, **kwargs):
    if start is None:
            start = np.zeros_like(end)
            
    # Validate input shapes
    if start.shape != end.shape:
        raise ValueError("start and end must have the same shape")
    
    # check dimensions of ends
    if len(end.shape) != 2 or end.shape[1] != 3:
        raise ValueError("ends must be an (1, 3) array-like")
    
    # Ensure thaat there is at least one arrow to plot
    if end.shape[0] == 0:
        raise ValueError("No arrows to plot; ends must not be empty")

    # if no color is provided, use black
    # if a single color string is provided, use it
    if color is None:
        arrow_color = 'k'
    else:
        arrow_color = color

    # Set default arrow properties
    # and update with any additional keyword arguments
    arrow_prop_dict = dict(
        mutation_scale=20, arrowstyle="-|>", shrinkA=0, shrinkB=0
    )
    arrow_prop_dict.update(kwargs)
    
    s = start[0]
    e = end[0]
    # create an Arrow3D object for the arrow
    a = Arrow3D(
        [s[0], e[0]],  # x coordinates of start and end
        [s[1], e[1]],  # y coordinates of start and end
        [s[2], e[2]],  # z coordinates of start and end
        label=label,
        color=arrow_color,
        **arrow_prop_dict,
    )
    ax.add_artist(a)

    # store starts/ends on the axes for setting the limits
    ax.points = np.vstack(
        (start, end, getattr(ax, "points", np.empty((0, 3))))
    )
    # only set limits if there are points to define them
    if ax.points.shape[0] > 0:
        for i, setter in enumerate(
            (ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d)
        ):
            min_val = ax.points[:, i].min()
            max_val = ax.points[:, i].max()

    return a

class Arrow3D(FancyArrowPatch):
    """
    A 3D arrow patch, used for plotting arrows in 3D space. Inherits from
    `matplotlib.patches.FancyArrowPatch` to leverage its functionality.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        Initializer of Arrow3D object.
        
        Parameters
        ----------
            xs, ys, zs : array-like
                The x, y, and z coordinates of the arrow's start and end points.
                
            *args, **kwargs : additional arguments
                Additional arguments are passed to the parent class
                `matplotlib.patches.FancyArrowPatch`.
        """
        # Initialize the base FancyArrowPatch with dummy start and end positions.
        super().__init__((0,0), (0,0), *args, **kwargs)
        # Store the 3D coordinates for later use in projection.
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """
        Projects the 3D arrow onto the 2D plane of the axes.
        
        Parameters
        ----------
            renderer : `~matplotlib.backend_bases.RendererBase`, default : None
                The renderer to use for the projection. If None, the current
                renderer is used.

        Returns
        -------
            float
                The minimum z-coordinate of the arrow in the projected space.
        """
        # Unpack the stored 3D coordinates.
        xs3d, ys3d, zs3d = self._verts3d
        
        # If the arrow is not associated with any axes, simply return the minimum z.
        if self.axes is None:
            return np.min(zs3d)
        
        # Use the proj3d module to convert 3D coordinates to 2D screen coordinates.
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        
        # Update the 2D positions of the FancyArrowPatch to the projected coordinates.
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        # Return the lowest z-value from the projection. This value is used to correctly order artists.
        return np.min(zs)

fig, ax = plt.subplots(1, 2, figsize=(12, 5),
                       subplot_kw=dict(projection="3d",
                                       proj_type='ortho'))

ax[0].voxels(np.ones((1, 1, 1)), facecolors=[0, 0, 0, 0], edgecolors='k')
# ax[0].add_artist(FancyArrow(0, 0, 0, 1, 1, 1))
endy = np.array([1,1,1])
ax[0] = arrows3d(end=endy.reshape(1,3), ax=ax[0], color='r')

ax[1].voxels(np.ones((1, 1, 1)), facecolors=[0, 0, 0, 0], edgecolors='k')
ax[1].view_init(elev=28, azim=45)
# ax[1].add_artist(FancyArrow(0, 0, 0, 1, 1, 1)))
ax[1] = arrows3d(end=endy.reshape(1,3), ax=ax[1], color='b')

plt.show()