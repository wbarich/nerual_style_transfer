from PIL import Image
import matplotlib.pyplot as plt
import ipdb
from nst import NST
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""read in the data"""
tubingen = Image.open('../images/tubingen.jpg').convert("RGB")
starry = Image.open('../images/starry-night.jpg').convert("RGB")
#kandinsky = Image.open('../images/kandinsky.jpg').convert("RGB")
#scream = Image.open('../images/edvard-munich-scream.jpg').convert("RGB")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Starry Night"""
styler = NST(tubingen, starry)
styler.artwork_image_size = 200
styler.max_iterations = 100
styler.learning_rate = 0.1
styler.alpha_beta_ratio = 1e-4
styler.style_weights = {"conv1_1" : 0.2, "conv2_1" : 0.2, "conv3_1" : 0.2, "conv4_1" : 0.2, "conv5_1" : 0.2}
styler.artwork_name = "Artwork s2"
styler.save = False
styler.incremental_plot = False
styler.create_artwork()
styler.print_frequency = int(styler.max_iterations/100)
ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""kandinsky @  1e-4"""
# styler = NST(tubingen, kandinsky)
# styler.artwork_image_size = 400
# styler.max_iterations = 10000
# styler.learning_rate = 0.001
# styler.alpha_beta_ratio = 1e-4
# styler.style_weights = {"conv1_1" : 0.2, "conv2_1" : 0.2, "conv3_1" : 0.2, "conv4_1" : 0.2, "conv5_1" : 0.2}
# styler.artwork_name = "Artwork 1e-4"
# styler.save = True
# styler.incremental_plot = False
# styler.print_frequency = styler.max_iterations/10
# styler.create_artwork()
# ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""kandinsky @  1e-3"""
# styler = NST(tubingen, kandinsky)
# styler.artwork_image_size = 400
# styler.max_iterations = 10000
# styler.learning_rate = 0.001
# styler.alpha_beta_ratio = 1e-3
# styler.style_weights = {"conv1_1" : 0.2, "conv2_1" : 0.2, "conv3_1" : 0.2, "conv4_1" : 0.2, "conv5_1" : 0.2}
# styler.artwork_name = "Artwork 1e-3"
# styler.save = True
# styler.incremental_plot = False
# styler.print_frequency = styler.max_iterations/10
# styler.create_artwork()
# ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Default Scream"""
# styler = NST(tubingen, scream)
# styler.artwork_image_size = 300
# styler.max_iterations = 10000
# styler.learning_rate = 0.001
# styler.alpha_beta_ratio = 1e-4
# styler.style_weights = {"conv1_1" : 0.2, "conv2_1" : 0.2, "conv3_1" : 0.2, "conv4_1" : 0.2, "conv5_1" : 0.2}
# styler.artwork_name = "Artwork 1e-4"
# styler.save = True
# styler.incremental_plot = False
# styler.print_frequency = styler.max_iterations/10
# styler.create_artwork()
# ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Default Scream weights 1-3-5-7-9"""
# styler = NST(tubingen, scream)
# styler.artwork_image_size = 300
# styler.max_iterations = 10000
# styler.learning_rate = 0.001
# styler.alpha_beta_ratio = 1e-4
# styler.style_weights = {"conv1_1" : 0.1, "conv2_1" : 0.3, "conv3_1" : 0.5, "conv4_1" : 0.7, "conv5_1" : 0.9}
# styler.artwork_name = "Artwork 1e-4"
# styler.save = True
# styler.incremental_plot = False
# styler.print_frequency = styler.max_iterations/10
# styler.create_artwork()
# ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Default Scream weights 1-3-5-7-9"""
# styler = NST(tubingen, scream)
# styler.artwork_image_size = 300
# styler.max_iterations = 10000
# styler.learning_rate = 0.001
# styler.alpha_beta_ratio = 1e-4
# styler.style_weights = {"conv1_1" : 0.1, "conv2_1" : 0.3, "conv3_1" : 0.5, "conv4_1" : 0.7, "conv5_1" : 0.9}
# styler.artwork_name = "Artwork 1e-4"
# styler.save = True
# styler.incremental_plot = False
# styler.print_frequency = styler.max_iterations/10
# styler.create_artwork()
# ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Default Scream weights 1-3-5-7-9"""
# styler = NST(tubingen, scream)
# styler.artwork_image_size = 300
# styler.max_iterations = 10000
# styler.learning_rate = 0.001
# styler.alpha_beta_ratio = 1e-4
# styler.style_weights = {"conv1_1" : 0.1, "conv2_1" : 0.3, "conv3_1" : 0.5, "conv4_1" : 0.7, "conv5_1" : 0.9}
# styler.artwork_name = "Artwork 1e-4"
# styler.save = True
# styler.incremental_plot = False
# styler.print_frequency = styler.max_iterations/10
# styler.create_artwork()
# ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Default Scream weights 1-3-5-7-9"""
# styler = NST(tubingen, scream)
# styler.artwork_image_size = 300
# styler.max_iterations = 10000
# styler.learning_rate = 0.001
# styler.alpha_beta_ratio = 1e-4
# styler.style_weights = {"conv1_1" : 0.1, "conv2_1" : 0.3, "conv3_1" : 0.5, "conv4_1" : 0.7, "conv5_1" : 0.9}
# styler.artwork_name = "Artwork 1e-4"
# styler.save = True
# styler.incremental_plot = False
# styler.print_frequency = styler.max_iterations/10
# styler.create_artwork()
# ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Default Scream weights 1-3-5-7-9"""
# styler = NST(tubingen, scream)
# styler.artwork_image_size = 300
# styler.max_iterations = 10000
# styler.learning_rate = 0.001
# styler.alpha_beta_ratio = 1e-4
# styler.style_weights = {"conv1_1" : 0.1, "conv2_1" : 0.3, "conv3_1" : 0.5, "conv4_1" : 0.7, "conv5_1" : 0.9}
# styler.artwork_name = "Artwork 1e-4"
# styler.save = True
# styler.incremental_plot = False
# styler.print_frequency = styler.max_iterations/10
# styler.create_artwork()
# ipdb.set_trace()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
