"""
=============
3D arrow plot
=============

Demonstrates plotting arrow in a 3D space. Here we plot two arrows from same start
point to different end points, and change the properties of the second arrow by passing
additional parameters other than the end point and start point to
`~matplotlib.patches.FancyArrowPatch`.

See `FancyArrowPatch documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html>`
for how the kwargs are being processed in FancyArrowPatch.
"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the start and end points of the arrow
start = np.array([0, 0, 0])
end = np.array([1, 1, 1])

# Create the arrow
ax.arrow3d(end, start)

end1 = np.array([1,2,3])
# changing the properties of the arrow and arrow is drawn from default start
# (0,0,0) to (1,2,3)
ax.arrow3d(end1, mutation_scale=20, color='r', arrowstyle='->', linewidth=2)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner
