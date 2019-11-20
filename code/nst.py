#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import ipdb

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class NST:

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __init__(self, content_image, style_image):

        #major attributes
        self.content_image = content_image #the content_image which has the content that we want the new image to have
        self.style_image = style_image #the content_image that has the style that we want to
        self.artwork_image_size = 300 #set the size of the artwork image

        #vgg attributes
        self.content_layers = ['conv4_2'] #list of all the layers from the cnn that we want to get the content information from
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'] #list of all the layers from the cnn that we want to get the style layers from
        self.cnn_map = {'0' : 'conv1_1', '5' : 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'} #map from conv name to element in vgg structure
        self.style_weights = {"conv1_1" : 0.2, "conv2_1" : 0.2, "conv3_1" : 0.2, "conv4_1" : 0.2, "conv5_1" : 0.2} #importance weighting for each layer in cnn style loss

        #algroithm attributes
        self.alpha_beta_ratio = 10e-3 #balance of weight of content to style
        self.alpha = 1 #weight of content loss in loss function
        self.beta = self.alpha/self.alpha_beta_ratio #weight of style loss in loss function
        self.learning_rate = 0.001 #rate at which the loss is updated according to gradient
        self.convergence_threshold = 10 #converge when error is less than this
        self.max_iterations = 500 #max iterations allowed by algorithm

        #plotting & debugging attributes
        self.debugging_messages = False #whether or not to print a message at each step in the process (for debugging)
        self.print_frequency = int(self.max_iterations/10) #after how many epochs we print update
        self.incremental_plot = False #save plot after print frequency
        self.artwork_name = 'artwork' #name of artwork that will be saved
        self.save = False #whether or not to save the result to disk

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_hardware_specs(self):

        """
        Ascertain whether there is a pytorch-usable GPU installed.
        """

        if torch.cuda.is_available():
            self.device = 'cuda'
            print('\n' + 'A GPU was detected.' + '\n')
        else:
            self.device = 'cpu'
            print('\n' + 'Warning - a GPU was not detected.' + '\n')

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def create_artwork(self):

        """
        Execute all methods necessary to style the content_image in the style of the style_image to produce artwork.
        """

        if self.debugging_messages:
            print('Starting ...')

        self.beta = self.alpha/self.alpha_beta_ratio #set beta in case it was updated

        self.get_hardware_specs() #test for gpu
        self.retrieve_vgg_network() #get vgg
        self.image_preprocessing() #prepare the images
        self.minimise_loss() #run algorithm
        self.convert_outputs() #convert from tensor to image
        self.save_plot(self.artwork, name = self.artwork_name, id = 'artwork', show = True) #save the result and save and show

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def convert_outputs(self):

        """
        Convert the outputs from tensor to image.
        """

        self.artwork = self.tensor_to_image(self.artwork, 'artwork')
        self.content = self.tensor_to_image(self.content_image, 'content_image')
        self.style_image = self.tensor_to_image(self.style_image, 'style_image')

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def retrieve_vgg_network(self):

        """
        Get the vgg trained model from pytorch.
        Set the requires_grad attribute to False to stop updates.
        """

        if self.debugging_messages:
            print('Now getting VGG network...')

        self.vgg = models.vgg19(pretrained = True).features
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False
        self.vgg.to(self.device)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def image_preprocessing(self):

        """
        Do some image pre-processing so that the methods can be run optimally.
        """

        if self.debugging_messages:
            print('Now transforming images...')

        process = transforms.Compose([transforms.Resize(self.artwork_image_size), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.content_image = process(self.content_image).type(torch.FloatTensor).to(self.device)
        self.style_image = process(self.style_image).type(torch.FloatTensor).to(self.device)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tensor_to_image(self, tensor, id):

        """
        Take an image tensor and return a numpy array [0, 1] that can be plotted in matplotlib.
        Used only for debugging/validation purposes.
        Inverse normalize is used becuase otherwise the images come out too dark. Following mehtod
        from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
        """

        if self.debugging_messages:
            print('Converting tensor to image...')

        image = tensor.cpu().clone().detach().numpy().squeeze() #make numpy
        image = image.transpose(1,2,0) #rearrange
        image = image*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5)) #scale values
        image = np.where(image > 1, 1, image)
        image = np.where(image < 0, 0, image)
        image = ((image - image.min())/(image.max() - image.min()) * 255).astype(np.uint8) #scale to 0-255
        return image

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_activation(self, tensor):

        """
        Using the cnn get the activation for a particular layer, for a particular input image, using a forward pass of the cnn.
        Used to calculate the content and style loss.

        params
        tensor: either the content or style image
        layers: which set of layers to look up (either content or style layers)

        returns
        feature_map: dict with activation of each layer that we are interested in.
        """

        if self.debugging_messages:
            print('Getting activation...')

        content_feature_maps = {}
        style_feature_maps = {}
        input_layer_values = tensor
        input_layer_values = input_layer_values.unsqueeze(0)
        for indx, cnn_layer in self.vgg._modules.items():
            input_layer_values = cnn_layer(input_layer_values)
            if indx in self.cnn_map:
                if self.cnn_map[indx] in self.content_layers:
                    content_feature_maps[self.cnn_map[indx]] = input_layer_values
                elif self.cnn_map[indx] in self.style_layers:
                    style_feature_maps[self.cnn_map[indx]] = input_layer_values

        return content_feature_maps, style_feature_maps

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_gram_matrix(self, feature_map):

        """
        Calculate the gram matrix for an input feature_map.
        """

        if self.debugging_messages:
            print('Getting gram matrix...')

        _, d, h, w = feature_map.size()
        feature_map = feature_map.view(d, h * w)
        style_gram = torch.mm(feature_map, feature_map.t())

        return style_gram

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_style_loss(self, artwork_style_feature_maps):

        """
        Calculate the style loss for a given gram matrix.
        Used in every epoch of the algorithm's update.
        """

        if self.debugging_messages:
            print('Getting style loss...')

        style_loss = 0

        for layer in self.style_layers:
            artwork_gram = self.get_gram_matrix(artwork_style_feature_maps[layer]) #get the gram matrix from the artwork activation layer
            style_loss += self.style_weights[layer] * torch.mean((self.style_target_grams[layer] - artwork_gram)**2) * (1/(4*(((self.artwork.shape[0] * self.artwork.shape[1])**2) * (9**2))))

        return style_loss

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def minimise_loss(self):

        """
        Run the algorithm that minimises the loss function.
        Call various methods in order to compute the different components of the loss function.
        """

        def sm(tensor):
            return np.abs(tensor.detach().cpu().numpy()).sum()

        if self.debugging_messages:
            print('Running main algorithm...')

        self.artwork = self.content_image.clone().requires_grad_(True).to(self.device) #create the data structure for the artwork
        self.get_targets() #initialise the content and style targets
        self.optimizer = torch.optim.Adam([self.artwork], lr = self.learning_rate) #set the optimisation module

        epoch = 1
        while True:

            #retrieve the feature maps from the cnn for the artwork
            artwork_content_feature_maps, artwork_style_feature_maps = self.get_activation(self.artwork) #get the feature_map representations

            #calculate the losses
            content_loss = torch.mean((self.content_feature_maps[self.content_layers[0]] - artwork_content_feature_maps[self.content_layers[0]])**2) #the content loss
            style_loss = self.get_style_loss(artwork_style_feature_maps) #the style loss
            total_loss = (self.alpha*content_loss) + (self.beta*style_loss) #the total loss

            #printing & saving incremental changes
            if epoch%self.print_frequency == 0:
                print("Epoch: " + str(epoch) + "| Loss: " + str(int(total_loss.item())) + " |" + str(int(epoch/self.max_iterations*100)) + "% complete.")
                if self.incremental_plot:
                    self.save_plot(self.artwork, self.artwork_name, id = 'artwork', show = False, epoch = epoch)

            #convergence test
            if (total_loss < self.convergence_threshold) or (epoch == self.max_iterations):
                break
            epoch += 1

            #update the artwork by gradient descent
            self.optimizer.zero_grad() #reset the gradients to avoid accumulation
            total_loss.backward() #gradient descent
            self.optimizer.step() #update params

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_targets(self):

        """
        Perform a forward pass with the content and feature images to get the feature maps in the cnns.
        Also compute the gram matricies of the style image, which is used as out target in the style loss
        calculations.
        These feature maps are used as the 'target' in our loss calculations.
        This is only called onces since the target isnt changing.
        """

        if self.debugging_messages:
            print('Setting the update targets...')

        self.content_feature_maps, _ = self.get_activation(self.content_image)
        _, self.style_feature_maps = self.get_activation(self.style_image)
        self.style_target_grams = {layer:self.get_gram_matrix(self.style_feature_maps[layer]) for layer in self.style_feature_maps}

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def show_comparison(self):

        """
        Show the result of the artwork with the original.
        """

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(self.style_image)
        ax1.title.set_text('Style Image')
        ax1.axis('off')
        ax2.imshow(self.content_image)
        ax2.title.set_text('Content Image')
        ax2.axis('off')
        ax3.imshow(self.artwork)
        ax3.title.set_text('Artwork')
        ax3.axis('off')
        plt.show()

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def save_plot(self, img, name, id, show = False, epoch = -1):

        """
        Take an image and save it to disk.
        """

        if not isinstance(img, np.ndarray):
            img = self.tensor_to_image(img, id)

        fig, ax1 = plt.subplots(1, 1)
        ax1.title.set_text(name)
        ax1.axis('off')
        ax1.imshow(img, label = name)
        if epoch != -1: #for the incremental plot save the plot to disk
            plt.imsave('../incremental plots/' + name + ' ' + str(epoch) + '.png', img, format='png')
            plt.close()
        elif self.save: #for the final result save to disk
            plt.imsave('../results/' + name + '.png', img, format='png')
            plt.show()
        else:
            plt.show()

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
