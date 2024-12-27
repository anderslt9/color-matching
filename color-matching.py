import cv2 as cv
from skimage import transform
from copy import deepcopy
import inspect
from random import shuffle
import numpy as np
import os

IMAGE_PATH = "imgs_out/"
PRECISION = .00001

def line_num():
  return inspect.currentframe().f_back.f_lineno

def err_print(*args):
  print(f"\033[93m{args[0]} at line {line_num()}\033[0m", *args[1:])

def float_to_int_img(img):
  return (img * 255).astype(int)

def check_same_val(val1, val2):
  if val1 < val2 + PRECISION and val1 > val2 - PRECISION: return True
  else: return False

class Image_Process:
  def __init__(self, img_path, name=None):
    self.base_img = cv.imread(img_path)
    self.reset_img()
    self.name = name
  
  def reset_img(self):
    """Resets image to base image before modifications
    """
    self.img = deepcopy(self.base_img)
  
  def reshape_img(self, reshape_size):
    """Reshapes image (saved as img) using reshape size which is either 
    - a tuple of integers to denote the desired size (ex. (64,128))
    - a tuple where one number is an integer denoting the desired size of that dimension and the other is -1 indicating to scale accordingly (ex. (64,-1))
    - a float < 1 is used to scale the image accordingly (ex. 0.5)

    Args:
        reshape_size (tuple[int,int]|float): reshapes image as specified
    """
    if type(reshape_size) == None: return
    
    # if float, resize by multiplying
    if type(reshape_size) != tuple: 
      if type(reshape_size) != float and type(reshape_size) != int: 
        err_print("Invalid input for reshape_size")
        return
      self.img = transform.resize(self.img, (round(self.img.shape[0] * reshape_size), round(self.img.shape[1] * reshape_size), 3), preserve_range=True).astype(int)
      self.img_size = self.img.shape[:2]
      return 
  
    # ensure reshape size is greater than 2 if tuple
    if len(reshape_size) < 2: 
      err_print("Tuple image size > 2")
      return
    
    # if reshape size has no negative number associated, resize directly 
    if reshape_size[0] > 0 and reshape_size[1] > 0: 
      self.img = transform.resize(self.img, (*reshape_size,3), preserve_range=True).astype(int)
      self.img_size = self.img.shape[:2]
      return
    
    # ensure that at least one number is positive
    if reshape_size[0] <= 0 and reshape_size[1] <= 0: 
      err_print("Both reshape_size parameters cannot be less than 0")
      return
    
    # resize negative size according to positive size
    if reshape_size[0] > 0: 
      ratio = reshape_size[0] / self.img.shape[0]
      self.img = transform.resize(self.img, (reshape_size[0], round(self.img.shape[1] * ratio), 3), preserve_range=True).astype(int)
    else:
      ratio = reshape_size[1] / self.img.shape[1]
      self.img = transform.resize(self.img, (round(self.img.shape[0] * ratio), self.img.shape[1], 3), preserve_range=True).astype(int)
    
    self.img_size = self.img.shape[:2]
  
  def display_img(self, img_name):
    if self.name != None: cv.imwrite(f'{IMAGE_PATH}{self.name}/{img_name}.jpg', self.img)
    else: cv.imwrite(f'{IMAGE_PATH}{img_name}.jpg', self.img)

class K_Means:
  def __init__(self, img):
    self.img = img
    self.pixels = img.reshape(-1,3)
    self.k_means = {}
    self.k_means_rounded = {}
    self.k_imgs = {}
  
  def distance_func(self, point1, point2):
    return np.linalg.norm(point1 - point2)
  
  def assign_pixels(self, k):
    if k not in self.k_means: 
      err_print(f"k={k} not computed before attempting to assign pixels")
      return self.img
    
    img = self.img
    new_img = np.zeros_like(img)
    k_means = self.k_means[k]
    k_means_rounded = self.k_means_rounded[k]
    
    for i in range(new_img.shape[0]):
      for j in range(new_img.shape[1]):
        pixel_dists = np.array([self.distance_func(mean, img[i,j]) for mean in k_means])
        min_dist_index = np.argmin(pixel_dists)
        new_img[i,j] = k_means_rounded[min_dist_index]
    
    self.k_imgs[k] = new_img
    return new_img
  
  def perform_k_means(self,k):
    pixels_copy = deepcopy(self.pixels)
    shuffle(pixels_copy)
    equal_group_val = len(pixels_copy) // k + 1
    sep_pixels = [pixels_copy[i:i+equal_group_val] for i in range(0, len(pixels_copy), equal_group_val)]
    prev_means = sorted([sum(pixel_group) / len(pixel_group) for pixel_group in sep_pixels], key=lambda a: a[0] * 65536 + a[1] * 256 + a[2])
    means = prev_means
    finished = False
    
    while not finished:
      finished = True
      clusters = [[] for _ in range(k)]
      
      for pixel in pixels_copy:
        pixel_dists = np.array([self.distance_func(prev_mean, pixel) for prev_mean in prev_means])
        min_dist_index = np.argmin(pixel_dists)
        clusters[min_dist_index].append(pixel)
      
      means = sorted([sum(cluster) / len(cluster) if len(cluster) != 0 else prev_means[i] for i,cluster in enumerate(clusters)], key=lambda a: a[0] * 65536 + a[1] * 256 + a[2])
      for mean, prev_mean in zip(means, prev_means):
        if False in [check_same_val(channel_mean, channel_mean_prev) for channel_mean, channel_mean_prev in zip(mean, prev_mean)]:
          finished = False
          prev_means = means
          break
    
    self.k_means[k] = means
    self.k_means_rounded[k] = [np.round(mean).astype(int) for mean in means]
    
if __name__ == "__main__":
  for img, name in zip(['imgs-test/ellie.jpg', 'imgs-test/spirit_happy.jpg', 'imgs-test/halloween.jpg', 'imgs-test/me_and_ellie.jpg', 'imgs-test/spirit_face.jpg'], ['ellie', 'spirit_happy', 'halloween', 'me_and_ellie', 'spirit_face']):
    if not os.path.isdir(f"imgs_out/{name}"): os.mkdir(f"imgs_out/{name}")
    for resolution_size in [32, 64, 128, 200]:
      img_proc = Image_Process(img, name)
      img_proc.reshape_img((resolution_size,-1))
      img_proc.display_img(f"{name}_{img_proc.img_size[0]}X{img_proc.img_size[1]}")
      k_means = K_Means(img_proc.img)
    
      for k in [3, 5, 10, 15, 20, 25, 30]:
        k_means.perform_k_means(k)
        new_img = k_means.assign_pixels(k)
        img_proc.img = new_img
        img_proc.display_img(f"{name}_modified_k{k}_{img_proc.img_size[0]}X{img_proc.img_size[1]}")
  
  