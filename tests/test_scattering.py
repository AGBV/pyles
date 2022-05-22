import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from pyles.functions.misc import multi2single_index
from pyles.functions.t_entry import t_entry
from pyles.functions.coupling_matrix_multiply import coupling_matrix_multiply
from pyles.functions.coupling_matrix_multiply import coupling_matrix_multiply_legacy
from pyles.functions.coupling_matrix_multiply import coupling_matrix_multiply_legacy_ab_free
from pyles.functions.coupling_matrix_multiply import coupling_matrix_multiply_numba

from pyles.particles import Particles
from pyles.initial_field import InitialField
from pyles.parameters import Parameters
from pyles.numerics import Numerics
from pyles.simulation import Simulation

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
  #                            focal_point=np.array((0,0,0)),
  #                            polar_angle=polar_angle,
  #                            azimuthal_angle=azimuthal_angle,
  #                            polarization='TE')
  #     parameters = Parameters(wavelength=wavelength,
  #                   medium_mefractive_index=medium_ref_idx,
  #                   particles=particles,
  #                   initial_field=initial_field)

  #     numerics = Numerics(lmax=lmax,
  #                   polar_angles=polar_angles,
  #                   azimuthal_angles=azimuthal_angles,
  #                   gpu=True,
  #                   particle_distance_resolution=1)

  #     simulation = Simulation(parameters, numerics)

  #     numerics.compute_translation_table()
  #     simulation.compute_mie_coefficients()
  #     simulation.compute_initial_field_coefficients()

  #     k_medium = complex(data['output']['scattering']['coupling'][0]['k_medium'])
  #     value = np.array(data['output']['scattering']['coupling'][0]['value'], dtype=complex)
  #     wx_test = np.array(data['output']['scattering']['coupling'][0]['Wx'], dtype=complex)
  #     wx = coupling_matrix_multiply_numba(simulation, value, k_medium=k_medium)
  #     np.testing.assert_allclose(wx, wx_test, relative_precision, relative_precision**2, True, 'Wx does not match.')
      

  #     # for coupling in data['output']['scattering']['coupling']:
  #     #   value = coupling['value']
  #     #   Wx = coupling_matrix_multiply(simulation, value)


if __name__ == '__main__':
  unittest.main()