import numpy as np

from pyles.particles import Particles
from pyles.initial_field import InitialField

class Parameters:

  def __init__(self, wavelength: np.array,
    medium_mefractive_index: np.array,
    particles: Particles,
    initial_field: InitialField):

    self.wavelength = wavelength
    self.medium_mefractive_index = medium_mefractive_index
    self.particles = particles
    self.initial_field = initial_field

    self.__setup()

  def __setup(self):
    self.__compute_omega()
    self.__compute_ks()

  def __compute_omega(self):
    self.omega = 2 * np.pi / self.wavelength

  def __compute_ks(self):
    self.k_medium = self.omega * self.medium_mefractive_index
    self.k_particle = np.outer(self.particles.m, self.omega)