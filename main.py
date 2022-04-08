import pandas as pd
import plotly.graph_objects as go

from particles import Particles
from pyles import Pyles

spheres = pd.read_csv('sphere_parameters.txt', names = ['x', 'y', 'z', 'r', 'n', 'k'])
particles = Particles(spheres.values[:,0:3], spheres.values[:,3], spheres.values[:,4:])
particles.compute_unique_refractive_indices()
# celes = Pyles(particles)

# celes.compute_mie_coefficients()

# fig = go.Figure(go.Scatter3d(
#   x = spheres.x,
#   y = spheres.y,
#   z = spheres.z,
#   mode = 'markers',
#   marker = dict(
#     sizemode = 'diameter',
#     sizeref = 10,
#     size = spheres.r * 2,
#     color = spheres.r,
#     opacity = 1
#   )
# ))
# fig.show()