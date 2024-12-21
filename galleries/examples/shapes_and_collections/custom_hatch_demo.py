"""
==========
Custom Hatchstyle demo
==========

Hatches can be added to most polygons in Matplotlib, including `~.Axes.bar`,
`~.Axes.fill_between`, `~.Axes.contourf`, and children of `~.patches.Polygon`.
They are currently supported in the PS, PDF, SVG, macosx, and Agg backends. The WX
and Cairo backends do not currently support hatching.
"""

import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()
hatchstyle = {
    "hatch": "-",
    "angle": 20,
    "scale": 8,
    "weight": 10,
}
rect = Rectangle((0.1, 0.1), 0.8, 0.8, hatchstyle=hatchstyle, facecolor="none")
rect._hatch_color = mpl.colors.to_rgba("orange")
ax.add_patch(rect)
plt.show()
plt.close()
plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.Rectangle`
#    - `matplotlib.axes.Axes.add_patch`
