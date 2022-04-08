import pandas as pd
import numpy as np

class Particles:
  def __init__(self, pos: np.array, r: np.array, m: np.array, type: str='sphere'):
    self.pos = pos
    self.r = r
    self.m = m
    self.type = type

  def compute_unique_refractive_indices(self):
    self.unique_refractive_indices = np.unique(self.m, axis=0)
    self.num_unique_refractive_indices = self.unique_refractive_indices.shape[0]
    print(self.num_unique_refractive_indices)