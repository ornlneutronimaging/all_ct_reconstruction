import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML
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

    def set_settings(self):

        title_ui = widgets.HTML("Select rotation angle (counter-clockwise)")
        self.angle_ui = widgets.RadioButtons(options=['-90 degrees', '0 degree', '+90 degrees'],
                                             value='-90 degrees',
                                            description='Angle')
        
        vbox = widgets.VBox([title_ui, self.angle_ui])
        display(vbox)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        image_rot_minus_90 = transform.rotate(self.parent.normalized_images[0], -90)
        image_normal = self.parent.normalized_images[0]
        image_rot_plut_90 = transform.rotate(self.parent.normalized_images[0], +90)

        axs[0].imshow(image_rot_minus_90, cmap='viridis', vmin=0, vmax=1)
        axs[0].set_title('-90 degrees')

        axs[1].imshow(image_normal, cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title('0 degree')

        axs[2].imshow(image_rot_plut_90, cmap='viridis', vmin=0, vmax=1)
        axs[2].set_title('+90 degrees')

    def _worker(self, _data, angle_value):
        data = transform.rotate(_data, angle_value)
        print(f"{np.shape(data) = }")
        print(f"{data = }")
        return data

    def apply_rotation(self):

        str_angle_value = self.angle_ui.value
        if str_angle_value == '-90 degrees':
            angle_value = -90
        elif str_angle_value == '0 degree':
            angle_value = 0
        else:
            angle_value = +90

        # worker_with_angle = partial(_worker, angle_value=angle_value)

        # logging.info(f"rotating the normalized_images by {angle_value} ...")        
        # with mp.Pool(processes=5) as pool:
        #      self.parent.normalized_images = pool.map(worker_with_angle, list(self.parent.normalized_images), angle_value)
    
        new_array_rotated = []
        for _data in tqdm(self.parent.normalized_images):
            new_array_rotated.append(transform.rotate(_data, angle_value, resize=True))

        self.parent.normalized_images = np.array(new_array_rotated)
        logging.info(f"rotating the normalized_images ... done!")        
