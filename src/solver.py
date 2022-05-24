import logging

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, bicgstab

class Solver:
  def __init__(self, solver_type: str='gmres', tolerance: float=1e-4, max_iter: int=1e4, restart: int=1e2):
    self.type       = solver_type.lower()
    self.tolerance  = tolerance
    self.max_iter   = max_iter
    self.restart    = restart

    self.log = logging.getLogger(__name__)

  def run(self, A: LinearOperator, b: np.ndarray, x0: np.ndarray=None):
    if x0 == None:
      x0 = np.copy(b)

    if self.type == 'bicgstab':
      value, info = bicgstab(A, b, x0, tol=self.tolerance, maxiter=self.max_iter)
    elif self.type == 'gmres':
      value, info = gmres(A, b, x0, restart=self.restart, tol=self.tolerance, atol=self.tolerance**2, maxiter=self.max_iter)
    else:
      self.log.error('Please specify a valid solver type')
      exit(1)

    return value, info
  #   function [value,convergenceHistory] = run(obj,mmm,rhs,varargin)
  #     if isempty(varargin)
  #         initial_guess = gather(rhs);
  #     else
  #         initial_guess = varargin{1};
  #     end
  #     if length(varargin) > 1
  #         verbose = varargin{2};
  #     else
  #         verbose = false;
  #     end

  #     prh = @(x) gather(obj.preconditioner.run(x,verbose));

  #     switch lower(obj.type)
  #         case 'bicgstab'
  #             [value,~,~,~,convergenceHistory] = ...
  #                 bicgstab_custom(mmm,rhs,obj.tolerance,obj.maxIter,prh,[],initial_guess);
  #         case 'gmres'
  #             fh = str2func('gmres');
  #             try
  #                 fetch_and_patch_gmres();
  #                 fh = str2func('gmres_monitor');
  #             catch
  #                 warning('using MATLAB default GMRES');
  #             end
  #             [value,~,~,~,convergenceHistory] = ...
  #                 fh(mmm,rhs,obj.restart,obj.tolerance,obj.maxIter,prh,[],initial_guess);
  #         otherwise
  #             error('please specify a valid solver type')
  #     end
  # end