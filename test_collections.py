import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"Matplotlib backend: {plt.get_backend()}")

try:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(f"Collections before adding: {len(ax.collections)}") # Should be 0

    # Create some line segments
    points = np.array([[[0,0,0], [1,1,1]], [[0,1,0], [1,0,1]]])
    lc = Line3DCollection(points)
    ax.add_collection(lc)

    print(f"Collections after adding Line3DCollection: {len(ax.collections)}") # Should be 1

    # Try adding a generic patch directly to see if it lands in collections
    from matplotlib.patches import Rectangle
    rect = Rectangle((0,0), 1, 1)
    ax.add_patch(rect) # In Axes3D, this should get wrapped into a Patch3DCollection

    print(f"Collections after adding Rectangle via add_patch: {len(ax.collections)}") # Should be 2 if successful

    plt.show() # Display the plot
    print("Script finished successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

