import ipywidgets as widgets
from IPython.display import display
from IPython.display import HTML
from ipywidgets import interactive
from tqdm import tqdm
import numpy as np
from skimage import transform
import multiprocessing as mp 
import logging
from functools import partial
import matplotlib.pyplot as plt

from __code.parent import Parent


def _worker(_data, angle_value):
    data = transform.rotate(_data, angle_value)
    print(f"{np.shape(data) = }")
    print(f"{data = }")
    return data


class Rotate(Parent):

    def is_rotation_needed(self):
        
        _, width = self.parent.normalized_images[0].shape[-2:]
        horizontal_center = width // 2

        display(HTML(f"<h3>The rotation axis of the sample must be VERTICAL! If it's not, you will need to rotate by 90degrees</h3>"))

        def plot_images(index):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            image = self.parent.normalized_images[index]
            ax.imshow(image, cmap='viridis', vmin=0, vmax=1)
            ax.axvline(x=horizontal_center, color='red', linestyle='--', label='Rotation Axis')
            ax.set_title(f"Image {index}")

        display_plot_images = interactive(plot_images, 
                                          index=widgets.IntSlider(min=0, 
                                                                  max=len(self.parent.normalized_images)-1, 
                                                                  step=1, 
                                                                  value=0))
        display(display_plot_images)

    def set_settings(self):
    
        title_ui = widgets.HTML("Select rotation angle")
        self.angle_ui = widgets.RadioButtons(options=['90 degrees', '0 degree'],
                                             value='90 degrees',
                                            description='Angle')
        
        vbox = widgets.VBox([title_ui, self.angle_ui])
        display(vbox)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        image_normal = self.parent.normalized_images[0]
        #image_rot_plut_90 = transform.rotate(self.parent.normalized_images[0], +90)
        image_rot_plus_90 = self.parent.normalized_images[0].swapaxes(-2, -1)[..., ::-1]

        axs[0].imshow(image_normal, cmap='viridis', vmin=0, vmax=1)
        axs[0].set_title('0 degree')

        axs[1].imshow(image_rot_plus_90, cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title('+90 degrees')

    def _worker(self, _data, angle_value):
        data = transform.rotate(_data, angle_value)
        return data

    def apply_rotation(self):

        logging.info("applying rotation ...")
        str_angle_value = self.angle_ui.value
        if str_angle_value == '90 degrees':
            angle_value = -90
        else:
            logging.info(f"not applying any rotation to the data!")
            return
    
        logging.info(f"\tangle_value = {angle_value}")

        # worker_with_angle = partial(_worker, angle_value=angle_value)

        # logging.info(f"rotating the normalized_images by {angle_value} ...")        
        # with mp.Pool(processes=5) as pool:
        #      self.parent.normalized_images = pool.map(worker_with_angle, list(self.parent.normalized_images), angle_value)
    
        logging.info(f"\tbefore rotation, {np.shape(self.parent.normalized_images) = }")
        new_array_rotated = self.parent.normalized_images.swapaxes(-2, -1)[..., ::-1]
        logging.info(f"\tafter rotation, {np.shape(new_array_rotated) = }")
        # new_array_rotated = []
        # for _data in tqdm(self.parent.normalized_images):
        #     new_array_rotated.append(transform.rotate(_data, angle_value, resize=True))

        self.parent.normalized_images = new_array_rotated[:]
        logging.info(f"rotating the normalized_images ... done!")        
