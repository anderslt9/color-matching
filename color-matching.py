import cv2 as cv
from skimage import transform
from copy import deepcopy
import inspect

IMAGE_PATH = "imgs_out/"

def line_num():
  return inspect.currentframe().f_back.f_lineno

def err_print(*args):
  print(f"\033[93m{args[0]} at line {line_num()}\033[0m", *args[1:])

def float_to_int_img(img):
  return (img * 255).astype(int)

class Image_Process:
  def __init__(self, img_path):
    self.base_img = cv.imread(img_path)
    self.reset_img()
  
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
      self.img = transform.resize(self.img, (int(self.img.shape[0] * reshape_size), int(self.img.shape[1] * reshape_size), 3), preserve_range=True).astype(int)
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
      self.img = transform.resize(self.img, (reshape_size[0], int(self.img.shape[1] * ratio), 3), preserve_range=True).astype(int)
    else:
      ratio = reshape_size[1] / self.img.shape[1]
      self.img = transform.resize(self.img, (int(self.img.shape[0] * ratio), self.img.shape[1], 3), preserve_range=True).astype(int)
    
    self.img_size = self.img.shape[:2]
  
  def display_img(self, img_name):
    cv.imwrite(f'{IMAGE_PATH}{img_name}.jpg', self.img)


if __name__ == "__main__":
  img_proc = Image_Process("imgs-test/spirit_happy.jpg")
  img_proc.reshape_img((64,-1))
  img_proc.display_img("spirit_happy_64X64")
  
  