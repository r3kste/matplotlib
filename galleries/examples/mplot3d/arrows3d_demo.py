# test_arrows.py
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Multiple arrow example
# You need to mention the `ends` parameter to specify where the arrows start.
# Optionally, you can also specify `starts` else it is taken as origin.
ends_data = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)

ax.arrows3d(
    ends_data,
    colors=["red", "green", "blue"],
    label="Multiple Arrows",
)

# You can also customize plot limits for better viewing
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

# set isometric view
ax.view_init(elev=35, azim=45)


ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Matplotlib 3D Arrows Test")
ax.legend()
plt.show()
# plt.close("all")
