import unittest
import glob
import re
import sys
from pathlib import Path

from src.solver import Solver
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.functions.misc import multi2single_index
from src.functions.t_entry import t_entry

from src.particles import Particles
from src.initial_field import InitialField
from src.parameters import Parameters
from src.numerics import Numerics
from src.simulation import Simulation

class TestScattering(unittest.TestCase):

  def test_t_entry_single_wavelength(self):
    # TODO
    # Generate new data with json
    # Use allclose to assert floating
    # Assert if the for loop hasn't been accessed at least ones !
    p = re.compile(r'mie_coefficients_lmax_(\d+)_wavelength_(\d+\.?\d*)_medium_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.csv')
    for data_file in glob.glob('tests/data/mie_coefficients_lmax*.csv'):

      res = p.search(data_file)
      lmax = int(res.group(1))
      wavelength = float(res.group(2))
      medium_real = float(res.group(3))
      medium_imag = float(res.group(4))
      medium = medium_real + 1j*medium_imag

      omega = 2 * np.pi / wavelength

      data = pd.read_csv(data_file, header=None).applymap(lambda val: complex(val.replace('i', 'j'))).to_numpy()
      radii     = np.real(data[:,0])
      ref_idx   = data[:,1]
      mie_test  = data[:,2:]

      mie = np.zeros(
        (data.shape[0],
        2 * lmax * (lmax + 2)),
        dtype=complex)
      
      for u_i in range(data.shape[0]):
        for tau in range(1, 3):
          for l in range(1, lmax+1):
            for m in range(-l,l+1):
              jmult = multi2single_index(0, tau, l, m, lmax)
              mie[u_i,jmult] = t_entry(tau=tau, l=l,
                k_medium = omega * medium,
                k_sphere = omega * ref_idx[u_i],
                radius = radii[u_i])

      np.testing.assert_array_almost_equal(mie,  mie_test,  decimal=5, err_msg='Mie coefficients do not match in %s' % (data_file), verbose=True)

  def test_t_entry_multiple_wavelength(self):
    # TODO
    # Generate new data with json
    # Use allclose to assert floating
    # Assert if the for loop hasn't been accessed at least ones !
    p = re.compile(r'mie_coefficients_.+_lmax_(\d+)_(.+)\.csv')
    for data_file in glob.glob('tests/data/mie_coefficients_*_lmax*.csv'):
      
      res = p.search(data_file)
      lmax = int(res.group(1))

      data = pd.read_csv(data_file, header=None).applymap(lambda val: complex(str(val).replace('i', 'j'))).to_numpy()

      num_particles = np.where(np.isnan(data[:,2]))[0][0]

      wavelength  = np.real(data[:,0])
      med_ref_idx = data[:,1]
      radii       = np.real(data[:num_particles,2])
      ref_idx     = data[:num_particles,3]
      mie_test    = np.reshape(data[:,4:].T, (num_particles, 2 * lmax * (lmax + 2), wavelength.size), order='F');

      omega = 2 * np.pi / wavelength

      mie = np.zeros(
        (num_particles,
        2 * lmax * (lmax + 2),
        wavelength.size),
        dtype=complex)
      
      for u_i in range(num_particles):
        for tau in range(1, 3):
          for l in range(1, lmax+1):
            for m in range(-l,l+1):
              jmult = multi2single_index(0, tau, l, m, lmax)
              mie[u_i,jmult,:] = t_entry(tau=tau, l=l,
                k_medium = omega * med_ref_idx,
                k_sphere = omega * ref_idx[u_i],
                radius = radii[u_i])

      np.testing.assert_array_almost_equal(mie,  mie_test,  decimal=5, err_msg='Mie coefficients do not match in %s' % (data_file), verbose=True)

  # def test_coupling_matrix(self):
  #   np.set_printoptions(precision=4, edgeitems=3, linewidth=np.inf, floatmode='fixed')
  #   executions = 0
  #   module = 'scatter'
  #   relative_precision = 1e-3
  #   p = re.compile(module + r'_(.+)(\.big)?\.json')
  #   for data_file in glob.glob('tests/data/%s_*.json' % module):

  #     # res = p.search(data_file)
  #     # identifier = res.group(1)
  #     # print(identifier)

  #     data = pd.read_json(data_file)
  #     lmax = data['input']['lmax']
  #     spheres = np.array(data['input']['particles'])
  #     wavelength = np.array(data['input']['wavelength'])
  #     medium_ref_idx = np.array([complex(x.replace('i', 'j')) for x in data['input']['medium_ref_idx']])

  #     polar_angle = data['input']['polar_angle']
  #     azimuthal_angle = data['input']['azimuthal_angle']
  #     polar_angles = np.array(data['input']['polar_angles'])
  #     azimuthal_angles = np.array(data['input']['azimuthal_angles'])

  #     particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])
  #     initial_field = InitialField(beam_width=0,
  #                                   focal_point=np.array((0,0,0)),
  #                                   polar_angle=polar_angle,
  #                                   azimuthal_angle=azimuthal_angle,
  #                                   polarization='TE')
  #     parameters = Parameters(wavelength=wavelength,
  #                             medium_mefractive_index=medium_ref_idx,
  #                             particles=particles,
  #                             initial_field=initial_field)

  #     solver = Solver(solver_type='gmres',
  #                     tolerance=5e-4,
  #                     max_iter=1e4,
  #                     restart=200)

  #     numerics = Numerics(lmax=lmax,
  #                         polar_angles=polar_angles,
  #                         azimuthal_angles=azimuthal_angles,
  #                         gpu=True,
  #                         particle_distance_resolution=1,
  #                         solver=solver)

  #     simulation = Simulation(parameters, numerics)

  #     numerics.compute_translation_table()
  #     simulation.compute_mie_coefficients()
  #     simulation.compute_initial_field_coefficients()
  #     simulation.compute_right_hand_side()

  #     value = np.array(data['output']['scattering']['coupling'][0]['value'], dtype=complex)
  #     wx_test = np.array(data['output']['scattering']['coupling'][0]['Wx'], dtype=complex)
  #     wx = simulation.coupling_matrix_multiply(value, idx=0)
  #     np.testing.assert_allclose(wx, wx_test, relative_precision, relative_precision**2, True, 'Wx does not match.')
      

      # for coupling in data['output']['scattering']['coupling']:
      #   value = coupling['value']
      #   Wx = coupling_matrix_multiply(simulation, value)

  def test_compute_right_hand_side(self):
    executions = 0
    module = 'precoupling'
    relative_precision = 1e-1
    for data_file in glob.glob('tests/data/simulation/%s_*.mat' % module):

      data = loadmat(data_file)

      lmax = int(data['lmax'][0][0])
      spheres = data['spheres']

      wavelength = data['wavelength'].squeeze()
      medium_ref_idx = data['medium_ref_idx'].squeeze()

      polar_angle = data['polar_angle'][0][0]
      azimuthal_angle = data['azimuthal_angle'][0][0]
      polar_angles = data['polar_angles'].squeeze()
      azimuthal_angles = data['azimuthal_angles'].squeeze()

      particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])
      initial_field = InitialField(beam_width=0,
                                    focal_point=np.array((0,0,0)),
                                    polar_angle=polar_angle,
                                    azimuthal_angle=azimuthal_angle,
                                    polarization='TE')
      parameters = Parameters(wavelength=wavelength,
                              medium_mefractive_index=medium_ref_idx,
                              particles=particles,
                              initial_field=initial_field)

      solver = Solver(solver_type='gmres',
                      tolerance=5e-1,
                      max_iter=1e3,
                      restart=200)

      numerics = Numerics(lmax=lmax,
                          polar_angles=polar_angles,
                          azimuthal_angles=azimuthal_angles,
                          gpu=True,
                          particle_distance_resolution=1,
                          solver=solver)

      simulation = Simulation(parameters, numerics)

      numerics.compute_translation_table()
      simulation.compute_mie_coefficients()
      simulation.compute_initial_field_coefficients()
      simulation.compute_right_hand_side()

      rhs = simulation.right_hand_side
      rhs_test = data['rhs']

      np.testing.assert_allclose(rhs, rhs_test, relative_precision, 0, True, 'The right hand side values do not match.')
      executions += 1

    self.assertGreater(executions, 0, 'No test data provided to be run.')


  # def test_coupling_matrix(self):
  #   np.set_printoptions(precision=4, edgeitems=3, linewidth=np.inf, suppress=True, formatter={
  #     'complexfloat': lambda x: '% +11.4e% +11.4ej ' % (x.real, x.imag)
  #   },floatmode='fixed')
  #   # import logging
  #   # logging.basicConfig(format='%(levelname)s (%(name)s): %(message)s', level=logging.INFO)
  #   executions = 0
  #   module = 'postcoupling'
  #   relative_precision = 1e-3
  #   for data_file in glob.glob('tests/data/simulation/%s_*.mat' % module):

  #     data = loadmat(data_file)

  #     lmax = int(data['lmax'][0][0])
  #     spheres = data['spheres']

  #     wavelength = data['wavelength'].squeeze()
  #     medium_ref_idx = data['medium_ref_idx'].squeeze()

  #     polar_angle = data['polar_angle'][0][0]
  #     azimuthal_angle = data['azimuthal_angle'][0][0]
  #     polarization = str(data['polarization'][0])
  #     beam_width = float(data['beamwidth'][0][0])

  #     polar_angles = data['polar_angles'].squeeze()
  #     azimuthal_angles = data['azimuthal_angles'].squeeze()

  #     solver_type = str(data['solver_type'][0])
  #     # solver_type = 'lgmres'
  #     tolerance = float(data['tolerance'][0][0])
  #     max_iter = int(data['max_iter'][0][0])
  #     restart = int(data['restart'][0][0])

  #     particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])
  #     initial_field = InitialField(beam_width=beam_width,
  #                                   focal_point=np.array((0,0,0)),
  #                                   polar_angle=polar_angle,
  #                                   azimuthal_angle=azimuthal_angle,
  #                                   polarization=polarization)

  #     parameters = Parameters(wavelength=wavelength,
  #                             medium_mefractive_index=medium_ref_idx,
  #                             particles=particles,
  #                             initial_field=initial_field)

  #     solver = Solver(solver_type=solver_type,
  #                     tolerance=tolerance,
  #                     max_iter=max_iter,
  #                     restart=restart)

  #     numerics = Numerics(lmax=lmax,
  #                         polar_angles=polar_angles,
  #                         azimuthal_angles=azimuthal_angles,
  #                         gpu=True,
  #                         particle_distance_resolution=1,
  #                         solver=solver)

  #     simulation = Simulation(parameters, numerics)

  #     numerics.compute_translation_table()
  #     simulation.compute_mie_coefficients()
  #     simulation.compute_initial_field_coefficients()
  #     simulation.compute_right_hand_side()

  #     # scattered_field_coefficients_test = data['sfc']

  #     # simulation.compute_scattered_field_coefficients()
  #     # simulation.compute_scattered_field_coefficients(guess=scattered_field_coefficients_test)
  #     # print(scattered_field_coefficients_test.ravel())
  #     # print(simulation.scattered_field_coefficients.ravel())

  #     # np.testing.assert_allclose(simulation.scattered_field_coefficients, scattered_field_coefficients_test, relative_precision, 0, True, 'The scattered field cofficients do not match.')

  #     scattering = loadmat('tests/data/simulation/scattering.mat')
  #     scattering = scattering['data'][0][0]
  #     value = scattering['value'].ravel(order='C')
  #     # value = scattering['value'].squeeze().reshape((simulation.parameters.particles.number, simulation.numerics.nmax), order='C').ravel(order='F')
  #     # wx = simulation.coupling_matrix_multiply(value, idx=0)
      
  #     # wx_test = scattering['Wx']
  #     # print(wx)
  #     # print(wx_test.ravel(order='C'))

  #     # twx = simulation.mie_coefficients[simulation.parameters.particles.single_unique_array_idx, :, 0].ravel(order='C') * wx
  #     # twx_test = scattering['TWx']
  #     # print(twx)
  #     # print(twx_test.ravel())

  #     mx = simulation.master_matrix_multiply(value, idx=0)
  #     mx_test = scattering['Mx']

  #     # np.testing.assert_allclose(twx.reshape((simulation.parameters.particles.number, simulation.numerics.nmax), order='F'), twx_test, relative_precision, 0, True, 'The test.')

  #     print('blub')
  #     print(mx)
  #     print(mx_test.ravel())
  #     # print(np.allclose(mx, mx_test, 1e-4, 1e-16))


if __name__ == '__main__':
  unittest.main()
