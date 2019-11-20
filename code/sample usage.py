#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from PIL import Image
import matplotlib.pyplot as plt
import ipdb
from nst import NST

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
This script is merely to demonstrate sample usage of the code.
"""

"""read in data - pls use pillow (PIL); not matplotlib (untested) to read in images"""
tubingen = Image.open('../images/tubingen.jpg').convert("RGB")
starry_night = Image.open('../images/starry-night.jpg').convert("RGB")

"""this generates the artwork"""
styler = NST(tubingen, starry_night)
styler.artwork_image_size = 100 #image size hugely affects training time. 100 is quick.
styler.create_artwork()
ipdb.set_trace() #merley here to keep artwork open
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
