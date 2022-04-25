import logging

import numpy as np

class InitialField:

  def __init__(self, beam_width, focal_point,
    field_type: str='gaussian', amplitude: float=1, 
    polar_angle: float=0, azimuthal_angle: float=0,
    polarization: str='TE'):

    self.field_type       = field_type
    self.amplitude        = amplitude
    self.polar_angle      = polar_angle
    self.azimuthal_angle  = azimuthal_angle
    self.polarization     = polarization
    self.beam_width       = beam_width
    self.focal_point      = focal_point

    self.__setup()
    self.log = logging.getLogger(__name__)

  def __set_pol_idx(self):
    match self.polarization.lower():
      case 'te':
        self.pol = 1
      case 'tm':
        self.pol = 2
      case _:
        self.pol = 1
        self.log.warning('{} is not a valid polarization type. Please use TE or TM. Reverting to TE'.format(self.polarization))

  def __set_normal_incidence(self):
    self.normal_incidence = np.abs(np.sin(self.polar_angle)) < 1e-5;

  def __setup(self):
    self.__set_pol_idx()
    self.__set_normal_incidence()