import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

class Particles:
  def __init__(self, pos: np.array, r: np.array, m: np.array, type: str='sphere'):
    self.pos = pos
    self.r = r
    self.m = m
    self.type = type

    self.number = r.shape[0]
    self.__setup_impl()

  def compute_unique_refractive_indices(self):
    self.unique_refractive_indices, self.refractive_index_array_idx = np.unique(
      self.m,
      return_inverse=True,
      axis=0)
    self.num_unique_refractive_indices = self.unique_refractive_indices.shape[0]

  def compute_unique_radii(self):
    self.unqiue_radii, self.radius_array_idx = np.unique(
      self.r,
      return_inverse=True,
      axis=0)
    self.num_unique_radii = self.unqiue_radii.shape[0]

  def compute_unique_radii_index_pairs(self):
    self.unique_radius_index_pairs, self.single_unique_array_idx = np.unique(
      np.column_stack((self.r, self.m)),
      return_inverse=True,
      axis=0)
    self.unique_single_radius_index_pairs = np.unique(
      np.column_stack((self.radius_array_idx, self.refractive_index_array_idx)), 
      axis=0)

  def compute_single_unique_idx(self):
    # self.single_unique_idx = np.multiply(
    #   np.sum(self.unique_single_radius_index_pairs, axis=1),
    #   np.sum(self.unique_single_radius_index_pairs, axis=1) + 1
    # ) // 2 + self.unique_single_radius_index_pairs[:,1]

    # pairedArray = np.multiply(
    #   self.radius_array_idx + self.refractive_index_array_idx,
    #   self.radius_array_idx + self.refractive_index_array_idx + 1
    # ) // 2 + self.refractive_index_array_idx

    # self.single_unique_idx, self.single_unique_array_idx = np.unique(
    #   pairedArray, 
    #   return_inverse=True, 
    #   axis=0)
    
    self.num_unique_pairs = self.unique_radius_index_pairs.shape[0]

  def compute_maximal_particle_distance(self):
    hull = ConvexHull(self.pos)
    vert = self.pos[hull.vertices, :]
    self.max_particle_distance = max(pdist(vert))
    print(self.max_particle_distance)

    

  def __setup_impl(self):
    self.compute_unique_refractive_indices()
    self.compute_unique_radii()
    self.compute_unique_radii_index_pairs()
    self.compute_single_unique_idx()
    self.compute_maximal_particle_distance()