import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.core.display import HTML
from ipywidgets import interactive
from IPython.display import display
import algotom.rec.reconstruction as rec
import numpy as np
import logging

from __code.parent import Parent
from __code.config import NUM_THREADS


class TestReconstruction(Parent):
    
    def select_slices(self):

        corrected_images_log = self.parent.corrected_images_log
        height, width = corrected_images_log[0].shape

        def plot_images(image_index, slice_1, slice_2):

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

            im = ax.imshow(corrected_images_log[image_index], cmap='viridis')
            plt.colorbar(im, ax=ax, shrink=0.5)

            ax.axhline(slice_1, color='red', linestyle='--', lw=2)
            ax.axhline(slice_2, color='red', linestyle='--', lw=2)

            plt.tight_layout()
            plt.show()

            return slice_1, slice_2

        self.display_plot = interactive(plot_images,
                                   image_index=widgets.IntSlider(min=0, max=len(corrected_images_log)-1, step=1, value=0),
                                   slice_1=widgets.IntSlider(min=0, max=height-1, step=1, value=500),
                                   slice_2=widgets.IntSlider(min=0, max=height-1, step=1, value=height-500))
        display(self.display_plot)

    def run_reconstruction(self):
        
        logging.info(f"Running reconstruction on 2 selected slices:")
        slice_1, slice_2 = self.display_plot.result
        self.parent.correct_images_log

        sinogram_corrected_images_log = np.moveaxis(self.parent.corrected_images_log, 0, 1)

        if self.parent.configuration.center_of_rotation == -1:
            center_of_rotation = np.shape(sinogram_corrected_images_log)[2] // 2
        else:
            center_of_rotation = self.parent.configuration.center_of_rotation

        reconstructed_slices = []
        for _slice in [slice_1, slice_2]:

            logging.info(f"\tworking with slice: {_slice}")
            _rec_img = rec.gridre_reconstruction(sinogram_corrected_images_log[_slice],
                                                 rot_center=center_of_rotation,
                                                 angles=self.parent.final_list_of_angles_rad,
                                                 apply_log=False,
                                                 ratio=1.0,
                                                 filter_name='shepp',
                                                 pad=100,
                                                 ncore=NUM_THREADS)
            reconstructed_slices.append(_rec_img)
            logging.info(f"\tdone!)
      
        