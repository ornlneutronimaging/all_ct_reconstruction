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

        normalized_images_log = self.parent.normalized_images_log
        height, width = normalized_images_log[0].shape
        max_value = 4

        def plot_images(image_index, slice_1, slice_2, vmin, vmax):

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

            im = ax.imshow(normalized_images_log[image_index], cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.5)

            ax.axhline(slice_1, color='red', linestyle='--', lw=2)
            ax.axhline(slice_2, color='red', linestyle='--', lw=2)

            plt.tight_layout()
            plt.show()

            return slice_1, slice_2

        self.display_plot = interactive(plot_images,
                                   image_index=widgets.IntSlider(min=0, 
                                                                 max=len(normalized_images_log)-1, 
                                                                 step=1, value=0, 
                                                                 layout=widgets.Layout(width='50%'),
                                                                 continuous_update=False),
                                   slice_1=widgets.IntSlider(min=0, 
                                                             max=height-1, 
                                                             step=1, 
                                                             value=400, 
                                                             layout=widgets.Layout(width='50%'),
                                                             continuous_update=False),
                                   slice_2=widgets.IntSlider(min=0, 
                                                             max=height-1, 
                                                             step=1, 
                                                             value=height-800, 
                                                             layout=widgets.Layout(width='50%'),
                                                             continuous_update=False),
                                   vmin=widgets.FloatSlider(min=0, 
                                                            max=max_value, 
                                                            step=0.01, 
                                                            value=0, 
                                                            layout=widgets.Layout(width='50%'),
                                                            continuous_update=False),
                                   vmax=widgets.FloatSlider(min=0, 
                                                            max=max_value, 
                                                            step=0.01, 
                                                            value=max_value, 
                                                            layout=widgets.Layout(width='50%'),
                                                            continuous_update=False),
                )
                                   
        display(self.display_plot)

    def run_reconstruction(self):
        
        logging.info(f"Running reconstruction on 2 selected slices:")
        slice_1, slice_2 = self.display_plot.result

        sinogram_normalized_images_log = np.moveaxis(self.parent.normalized_images_log, 0, 1)

        if self.parent.configuration.center_of_rotation == -1:
            center_of_rotation = np.shape(sinogram_normalized_images_log)[2] // 2
        else:
            center_of_rotation = self.parent.configuration.center_of_rotation

        # reconstructed_slices = []
        for _slice in [slice_1, slice_2]:

            logging.info(f"\tworking with slice: {_slice}")

            # logging.info(f"\tusing rec.gridrec_reconstruction")
            # _rec_img = rec.gridrec_reconstruction(sinogram_normalized_images_log[_slice],
            #                                      center_of_rotation,
            #                                      angles=self.parent.final_list_of_angles_rad,
            #                                      apply_log=False,
            #                                      ratio=1.0,
            #                                      filter_name='shepp',
            #                                      pad=100,
            #                                      ncore=NUM_THREADS)    
            
            logging.info(f"\tusing rec.astra_reconstruction")
            _rec_img = rec.astra_reconstruction(sinogram_normalized_images_log[_slice],
                                                 center_of_rotation,
                                                 angles=self.parent.final_list_of_angles_rad,
                                                 apply_log=False,
                                                 ratio=1.0,
                                                 filter_name='hann',
                                                 pad=None,
                                                 num_iter=300,
                                                 method='SIRT_CUDA',
                                                 ncore=NUM_THREADS)


            logging.info(f"\t{np.shape(_rec_img) =}")
            logging.info(f"\tslice: {_slice}")

            self.display_reconstructed_slice(_rec_img, _slice)

        logging.info(f"\tdone!")
      
    def display_reconstructed_slice(self, reconstructed_slice, slice_number):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

        im0 = ax.imshow(reconstructed_slice, cmap='viridis', vmin=0)
        plt.colorbar(im0, ax=ax, shrink=0.5)
        ax.set_title(f"Slice: {slice_number}")
        ax.axis('off')

        plt.tight_layout()
        plt.show()
