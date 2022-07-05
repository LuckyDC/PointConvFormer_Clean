import random 

import numpy as np 
import scipy
import scipy.ndimage 
import scipy.interpolate 
from numpy import cross
from scipy.linalg import expm, norm
import torch

class Compose(object):
  """Composes several transforms together."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, *args):
    for t in self.transforms:
      args = t(*args)
    return args

class RandomDropColor(object):
    def __init__(self, p=0.8, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment
    
    def __call__(self, coords, color, labels, norms):
        t = torch.rand(1).numpy()[0]
        # print(t)
        if color is not None and t > self.p:
            color *= self.color_augment
        return coords, color, labels, norms

    def __repr__(self):
        return 'RandomDropColor(color_augment: {}, p: {})'.format(self.color_augment, self.p)


class RandomDropout(object):

  def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.dropout_ratio = dropout_ratio
    self.dropout_application_ratio = dropout_application_ratio

  def __call__(self, coords, feats, labels, norms):
    if random.random() < self.dropout_ratio:
      N = len(coords)
      inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
      return coords[inds], feats[inds], labels[inds], norms[inds]
    return coords, feats, labels, norms 

class RandomHorizontalFlip(object):

  def __init__(self, upright_axis):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.D = 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, coords, feats, labels, norms):
    if random.random() < 0.95:
      for curr_ax in self.horz_axes:
        if random.random() < 0.5:
          coords[:, curr_ax] = - coords[:, curr_ax]
          norms[:, curr_ax] = - norms[:, curr_ax]
    return coords, feats, labels, norms 

class ChromaticTranslation(object):
  """Add random color to the image, input must be an array in [0,255] or a PIL image"""

  def __init__(self, trans_range_ratio=1e-1):
    """
    trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
    """
    self.trans_range_ratio = trans_range_ratio

  def __call__(self, coords, feats, labels, norms):
    if torch.rand(1).numpy()[0] < 0.95:
      tr = (torch.rand(1, 3).numpy() - 0.5) * 255 * 2 * self.trans_range_ratio
      feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
    return coords, feats, labels, norms 

class ChromaticAutoContrast(object):

  def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
    self.randomize_blend_factor = randomize_blend_factor
    self.blend_factor = blend_factor

  def __call__(self, coords, feats, labels, norms):
    if torch.rand(1).numpy()[0] < 0.2:
      # mean = np.mean(feats, 0, keepdims=True)
      # std = np.std(feats, 0, keepdims=True)
      # lo = mean - std
      # hi = mean + std
      lo = np.min(feats, 0, keepdims=True)
      hi = np.max(feats, 0, keepdims=True)

      scale = 255 / (hi - lo)

      contrast_feats = (feats - lo) * scale

      blend_factor = torch.rand(1).numpy()[0] if self.randomize_blend_factor else self.blend_factor
      feats = (1 - blend_factor) * feats + blend_factor * contrast_feats
    return coords, feats, labels, norms 


class ChromaticJitter(object):

  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, coords, feats, labels, norms):
    if torch.rand(1).numpy()[0] < 0.95:
      # noise = np.random.randn(feats.shape[0], 3)
      noise = torch.randn(feats.shape[0], 3).numpy()
      noise *= self.std * 255
      feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
    return coords, feats, labels, norms

def elastic_distortion(pointcloud, granularity, magnitude):
  """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
  blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
  blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
  blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
  coords = pointcloud[:, :3]
  coords_min = coords.min(0)

  # Create Gaussian noise tensor of the size given by granularity.
  noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
  noise = np.random.randn(*noise_dim, 3).astype(np.float32)

  # Smoothing.
  for _ in range(2):
    noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
    noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
    noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

  # Trilinear interpolate noise filters for each spatial dimensions.
  ax = [
      np.linspace(d_min, d_max, d)
      for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                 (noise_dim - 2), noise_dim)
  ]
  interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
  pointcloud[:, :3] = coords + interp(coords) * magnitude
  return pointcloud

# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(cross(np.eye(3), axis / norm(axis) * theta))

LOCFEAT_IDX = 2

def get_transformation_matrix(rotation_augmentation_bound, scale_augmentation_bound, rotation_angle = None):
    scale_matrix = np.eye(4)
    rotation_matrix = np.eye(4)

    # Random rotation
    rot_mat = np.eye(3)

    rot_mats = []
    for axis_ind, rot_bound in enumerate(rotation_augmentation_bound):
      theta = 0
      axis = np.zeros(3)
      axis[axis_ind] = 1
      if rot_bound is not None:
        theta = np.random.uniform(*rot_bound)
      rot_mats.append(M(axis, theta))

    # Use random order
    np.random.shuffle(rot_mats)
    rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]

    if rotation_angle is not None:
      axis = np.zeros(3)
      axis[LOCFEAT_IDX] = 1
      rot_mat = M(axis, rotation_angle)

    rotation_matrix[:3, :3] = rot_mat 

    # Scale 
    scale = np.random.uniform(scale_augmentation_bound)
    np.fill_diagonal(scale_matrix[:3, :3], scale)
    return scale_matrix, rotation_matrix
