from time import time
import logging

import pandas as pd
import numpy as np
# import plotly.graph_objects as go
# from functions.coupling_matrix_multiply import coupling_matrix_multiply

from src.particles import Particles
from src.initial_field import InitialField
from src.parameters import Parameters
from src.numerics import Numerics
from src.simulation import Simulation

from src.functions.misc import jmult_max

logging.basicConfig(format='%(levelname)s (%(name)s): %(message)s', level=logging.INFO)

spheres = pd.read_csv('data/sphere_parameters.csv', names = ['x', 'y', 'z', 'r', 'n', 'k'])
wavelength = pd.read_csv('data/lambda.csv', header=None).to_numpy()
lmax = 4

mie_coefficients = pd.read_csv('data/test/mie_coefficients.csv', header=None).applymap(lambda val: complex(val.replace('i', 'j'))).to_numpy()
translation_ab5_csv = pd.read_csv('data/test/translation_ab5.csv', header=None, dtype=str).applymap(lambda val: complex(val.replace('i', 'j'))).to_numpy()
translation_ab5 = np.zeros((jmult_max(1, lmax), jmult_max(1, lmax), 2*lmax+1), dtype=complex)
for k in range(translation_ab5_csv.shape[0]):
  x = int(np.real(translation_ab5_csv[k, 0]))
  y = int(np.real(translation_ab5_csv[k, 1]))
  z = int(np.real(translation_ab5_csv[k, 2]))
  translation_ab5[x, y, z] = translation_ab5_csv[k, 3]

particles = Particles(spheres.values[:,0:3], spheres.values[:,3], spheres.values[:,4:])
initial_field = InitialField(beam_width=0,
                             focal_point=np.array((0,0,0)),
                             polar_angle=0,
                             azimuthal_angle=0,
                             polarization='TE')

wavelength = np.array([200])
medium_mefractive_index_scalar = 1
inputs = Parameters(wavelength=wavelength,
                    medium_mefractive_index=np.ones(wavelength.shape) * medium_mefractive_index_scalar,
                    particles=particles,
                    initial_field=initial_field)
numerics = Numerics(lmax=lmax,
                    polar_angles=np.arange(0, np.pi * (1 + 1/5e3), np.pi / 5e3),
                    azimuthal_angles=np.arange(0, np.pi * (2 + 1/1e2), np.pi / 1e2),
                    particle_distance_resolution=1,
                    gpu=False)

simulation = Simulation(inputs, numerics)
simulation.compute_mie_coefficients()
simulation.compute_translation_table()

#simulation.compute_initial_field_coefficients()
#print(simulation.initial_field_coefficients)

# coupling_matrix_multiply()
# fig = go.Figure()
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
# fig.show()
