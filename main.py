import pandas as pd
import plotly.graph_objects as go

from particles import Particles
from pyles import Pyles

spheres = pd.read_csv('sphere_parameters.txt', names = ['x', 'y', 'z', 'r', 'n', 'k'])
particles = Particles(spheres.values[:,0:3], spheres.values[:,3], spheres.values[:,4:])

# fig = go.Figure()
# fig.add_trace(go.Scatter3d(
#   x = particles.hull[:,0],
#   y = particles.hull[:,1],
#   z = particles.hull[:,2],
#   mode = 'markers',
#   marker = dict(
#     color = "red",
#     opacity = 1
#   )
# ))
# fig.add_trace(go.Scatter3d(
#   x = spheres.x,
#   y = spheres.y,
#   z = spheres.z,
#   mode = 'markers',
#   marker = dict(
#     sizemode = 'diameter',
#     sizeref = 10,
#     size = spheres.r * 2,
#     color = spheres.r,
#     opacity = 0.5
#   )
# ))
# fig.show()