import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from pyles.particles import Particles

class TestParticles(unittest.TestCase):

  def test_compute_unique_refractive_indices(self):

    p = re.compile(r'compute_unique_refractive_indices_(.+)\.csv')
    for data_file in glob.glob('tests/data/particles/compute_unique_refractive_indices_*.csv'):

      res = p.search(data_file)
      input_file = res.group(1)
      
      spheres = pd.read_csv('tests/input/%s.csv' % input_file, header=None)
      particles = Particles(spheres.values[:,0:3], spheres.values[:,3], spheres.values[:,4:])

      data = pd.read_csv(data_file, header=None)
      num_unique_refractive_indices = int(data.values[0,0])
      unique_refractive_indices = np.array([complex(str(x).replace('i', 'j')) for x in data.loc[1:num_unique_refractive_indices,0]])
      refractive_index_array_idx = np.array(data[0][num_unique_refractive_indices+1:])

      self.assertEqual(particles.num_unique_refractive_indices, num_unique_refractive_indices, 'Number of unique refractive indices is not equal.')
      np.testing.assert_array_equal(particles.unique_refractive_indices, unique_refractive_indices, 'Unique refractive indices do not matchh.')
      np.testing.assert_array_equal(particles.refractive_index_array_idx, refractive_index_array_idx, 'Refractive index array indices do not match.')

  def test_compute_unique_radii(self):

    p = re.compile(r'compute_unique_radii_(.+)\.csv')
    for data_file in glob.glob('tests/data/particles/compute_unique_radii_[!index]*.csv'):

      res = p.search(data_file)
      input_file = res.group(1)
      
      spheres = pd.read_csv('tests/input/%s.csv' % input_file, header=None)
      particles = Particles(spheres.values[:,0:3], spheres.values[:,3], spheres.values[:,4:])

      data = pd.read_csv(data_file, header=None)
      num_unique_radii = int(data.values[0,0])
      unqiue_radii = np.squeeze(data[:][1:num_unique_radii+1].applymap(lambda val: complex(str(val).replace('i', 'j'))).to_numpy())

      self.assertEqual(particles.num_unique_radii, num_unique_radii, 'Number of unique radii is not equal.')
      np.testing.assert_array_equal(particles.unqiue_radii, unqiue_radii, 'Unique radii do not matchh.')

  def test_compute_unique_radii_index_pairs(self):

    p = re.compile(r'compute_unique_radii_index_pairs_(.+)\.csv')
    for data_file in glob.glob('tests/data/particles/compute_unique_radii_index_pairs_*.csv'):

      res = p.search(data_file)
      input_file = res.group(1)
      
      spheres = pd.read_csv('tests/input/%s.csv' % input_file, header=None)
      particles = Particles(spheres.values[:,0:3], spheres.values[:,3], spheres.values[:,4:])

      data = pd.read_csv(data_file, header=None)
      num_unique_radii_index_pairs = int(data.values[0,0])
      unique_radius_index_pairs = data[:][1:num_unique_radii_index_pairs+1].applymap(lambda val: complex(str(val).replace('i', 'j'))).to_numpy()
      unique_single_radius_index_pairs = data[:][num_unique_radii_index_pairs+1:2*num_unique_radii_index_pairs+1].applymap(lambda val: int(val)).to_numpy()
      radius_array_idx = np.array(data[0][2*num_unique_radii_index_pairs+1:].astype(int))

      self.assertEqual(particles.unique_radius_index_pairs.shape[0], num_unique_radii_index_pairs, 'Number of unique radius-refractive index pairs is not equal.')
      np.testing.assert_array_equal(particles.unique_radius_index_pairs, unique_radius_index_pairs, 'Unique radius-refractive index pairs do not match.')
      np.testing.assert_array_equal(particles.unique_single_radius_index_pairs, unique_single_radius_index_pairs, 'Unique single radius-refractive index pairs do not match.')
      np.testing.assert_array_equal(particles.radius_array_idx, radius_array_idx, 'Radius array indices do not match.')

  def test_compute_single_unique_idx(self):

    p = re.compile(r'compute_single_unique_idx_(.+)\.csv')
    for data_file in glob.glob('tests/data/particles/compute_single_unique_idx_*.csv'):

      res = p.search(data_file)
      input_file = res.group(1)
      
      spheres = pd.read_csv('tests/input/%s.csv' % input_file, header=None)
      particles = Particles(spheres.values[:,0:3], spheres.values[:,3], spheres.values[:,4:])

      data = pd.read_csv(data_file, header=None)
      num_unique_pairs = int(data.values[0,0])
      single_unique_idx = np.squeeze(data[:][1:num_unique_pairs+1].applymap(lambda val: int(val)).to_numpy())
      single_unique_array_idx = np.squeeze(data[:][num_unique_pairs+1:].applymap(lambda val: int(val)).to_numpy())
      _, idx1 = np.unique(single_unique_array_idx, return_inverse=True)
      _, idx2 = np.unique(particles.single_unique_array_idx, return_inverse=True)

      self.assertEqual(particles.num_unique_pairs, num_unique_pairs, 'Number of unique pairs is not equal.')
      # np.testing.assert_array_equal(particles.single_unique_idx, single_unique_idx, 'Unique radius-refractive index pairs do not match.')
      np.testing.assert_array_equal(idx1, idx2, 'Unique single array indices do not match.')

  def test_compute_maximal_particle_distance(self):

    p = re.compile(r'compute_maximal_particle_distance_(.+)\.csv')
    for data_file in glob.glob('tests/data/particles/compute_maximal_particle_distance_*.csv'):

      res = p.search(data_file)
      input_file = res.group(1)
      
      spheres = pd.read_csv('tests/input/%s.csv' % input_file, header=None)
      particles = Particles(spheres.values[:,0:3], spheres.values[:,3], spheres.values[:,4:])

      data = pd.read_csv(data_file, header=None)
      max_particle_distance = float(data.values[0,0])

      self.assertAlmostEqual(particles.max_particle_distance, max_particle_distance, 10, 'Max particle distance is not equal.')


if __name__ == '__main__':
  unittest.main()