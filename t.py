import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

num_points_x = 10
num_points_y = 9
x = np.linspace(0, 1, num_points_x)
y = np.linspace(0, 1, num_points_y)

X, Y = np.meshgrid(x, y)
X[1::2, :] += (x[1] - x[0]) / 2  # stagger every alternate row

col = ax.scatter(
    X.ravel(),
    Y.ravel(),
    s=1700,
    facecolor="none",
    edgecolor="gray",
    linewidth=2,
    marker="h",  # Use hexagon as marker
    hatch="xxx",
    hatchcolor="tab:blue",
)

hatch_linewidths = []
for linewidth in np.linspace(0.5, 4, num_points_y):
    hatch_linewidths.extend([linewidth] * num_points_x)

col._hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
col._hatch_linewidths = hatch_linewidths
col._alphas = np.linspace(1, 0.1, num_points_x)
col._forced_alphas = [True]
col._joinstyles = ["miter"]
col._capstyles = ["round"]

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.savefig("t.svg")
