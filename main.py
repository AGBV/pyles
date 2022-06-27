#%% Import libs and set logging
from time import time
import logging

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.particles import Particles
from src.initial_field import InitialField
from src.parameters import Parameters
from src.solver import Solver
from src.numerics import Numerics
from src.simulation import Simulation
from src.optics import Optics

# logging.basicConfig(format='%(levelname)s (%(name)s): %(message)s', level=logging.INFO)

#%% Set imputs
spheres = pd.read_csv('data/sphere_parameters.csv', names = ['x', 'y', 'z', 'r', 'n', 'k']).to_numpy()
spheres[:,5] = spheres[:,4] / 10
wavelength = pd.read_csv('data/lambda.csv', header=None).to_numpy().squeeze()
wavelength = wavelength[6::10]

medium_mefractive_index = np.ones_like(wavelength) * 1.63
lmax = 4

#%% Set up all objects
particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])

initial_field = InitialField(beam_width=0,
                             focal_point=np.array((0,0,0)),
                             polar_angle=0,
                             azimuthal_angle=0,
                             polarization='UNP')

parameters = Parameters(wavelength=wavelength,
                    medium_mefractive_index=medium_mefractive_index,
                    particles=particles,
                    initial_field=initial_field)

solver = Solver(solver_type='lgmres',
                tolerance=5e-4,
                max_iter=1000,
                restart=500)

numerics = Numerics(lmax=lmax,
                    sampling_points_number=[360, 180],
                    particle_distance_resolution=1,
                    gpu=True,
                    solver=solver)

simulation = Simulation(parameters, numerics)

optics = Optics(simulation)


#%% Run needed calculations
particles.compute_volume_equivalent_area()
numerics.compute_spherical_unity_vectors()
numerics.compute_translation_table()
simulation.compute_mie_coefficients()
simulation.compute_initial_field_coefficients()
simulation.compute_right_hand_side()
simulation.compute_scattered_field_coefficients()
optics.compute_cross_sections()
optics.compute_phase_funcition()
print('Done')

#%% Display values
import plotly.io as pio
png_renderer = pio.renderers['vscode']

fig = make_subplots(rows=2, cols=2,
  subplot_titles=('Extinction Cross Section', 'Scattering Cross Section', 'Albedo', 'Phase Function'),
  specs=[[{'type': 'xy'}, {'type': 'xy'}],
         [{'type': 'xy'}, {'type': 'polar'}]])

fig.add_trace(
  go.Scatter(
    x = wavelength/1e3,
    y = optics.c_ext,
    mode = 'lines',
    name = 'C ext'
  ), row = 1, col = 1
)

fig.add_trace(
  go.Scatter(
    x = wavelength/1e3,
    y = optics.c_sca,
    mode = 'lines',
    name = 'C sca'
  ), row = 1, col = 2
)

fig.add_trace(
  go.Scatter(
    x = wavelength/1e3,
    y = optics.albedo,
    mode = 'lines',
    name = 'w'
  ), row = 2, col = 1
)

for w in range(len(wavelength)):
  fig.add_trace(
    go.Scatterpolar(
      theta = np.rad2deg(optics.scattering_angles),
      r = optics.phase_function[:, w],
      mode = 'lines',
      name = '%.2f&mu;m' % (wavelength[w] / 1e3),
      subplot = 'polar'
    ), row = 2, col = 2
  )


fig.update_layout(
  xaxis1 = dict(
    title='Wavelength',
    ticksuffix='&mu;m'
  ),
  yaxis1 = dict(
    title = 'C ext'
  ),
  xaxis2 = dict(
    title = 'Wavelength',
    ticksuffix = '&mu;m'
  ),
  yaxis2 = dict(
    title = 'C sca'
  ),
  xaxis3 = dict(
    title = 'Wavelength',
    ticksuffix = '&mu;m'
  ),
  yaxis3 = dict(
    title = 'Albedo'
  ),
  polar = dict(
    radialaxis = dict(type = 'log'),
    sector = [0, 180]
  )
)

fig.write_image('main.png')
# fig.show()

# %%
