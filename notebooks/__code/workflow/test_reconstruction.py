import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.core.display import HTML
from ipywidgets import interactive
from IPython.display import display
import algotom.rec.reconstruction as rec
import numpy as np
import logging
import svmbir

from __code.parent import Parent
from __code.config import svmbir_parameters
from __code.config import NUM_THREADS, SVMBIR_LIB_PATH


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
        logging.info(f"\t{np.shape(sinogram_normalized_images_log) = }")
        logging.info(f"\t{np.shape(self.parent.normalized_images_loglog) = }")
        logging.info(f"\t{np.shape(self.parent.final_list_of_angles_rad) = }")

        if self.parent.configuration.center_of_rotation == -1:
            center_of_rotation = np.shape(sinogram_normalized_images_log)[2] // 2
        else:
            center_of_rotation = self.parent.configuration.center_of_rotation

        nbr_angles, height, width = np.shape(self.parent.normalized_images_log)  # width of the image
        center_offset = -(width / 2 - center_of_rotation)  # it's Shimin's formula

        # reconstructed_slices = []
        for _slice in [slice_1, slice_2]:

            logging.info(f"\tworking with slice: {_slice}")

            logging.info(f"\tusing rec.gridrec_reconstruction")
            _rec_img_gridrec = rec.gridrec_reconstruction(sinogram_normalized_images_log[_slice],
                                                          center_of_rotation,
                                                          angles=self.parent.final_list_of_angles_rad,
                                                          apply_log=False,
                                                          ratio=1.0,
                                                          filter_name='shepp',
                                                          pad=100,
                                                          ncore=NUM_THREADS)    
            logging.info(f"\t{np.shape(_rec_img_gridrec) =}")
            logging.info(f"\tusing rec.gridrec_reconstruction ... done!")
            
            logging.info(f"\tusing rec.astra_reconstruction")
            _rec_img_astra = rec.astra_reconstruction(sinogram_normalized_images_log[_slice],
                                                      center_of_rotation,
                                                      angles=self.parent.final_list_of_angles_rad,
                                                      apply_log=False,
                                                      ratio=1.0,
                                                      filter_name='hann',
                                                      pad=None,
                                                      num_iter=300,
                                                      method='SIRT_CUDA',
                                                      ncore=NUM_THREADS)
            logging.info(f"\t{np.shape(_rec_img_astra) =}")
            logging.info(f"\tusing rec.astra_reconstruction ... done!")

            logging.info(f"\tusing rec.svmbir_reconstruction")
            projections_normalized_images_log = self.parent.normalized_images_log[:, _slice:_slice+1, :]
            logging.info(f"\t{np.shape(projections_normalized_images_log) = }")
            logging.info(f"\t{np.shape(self.parent.final_list_of_angles_rad) =}")
            logging.info(f"\t{height = }")
            logging.info(f"\t{width = }")
            logging.info(f"\t{center_offset = }")

            _rec_img_svmbir = svmbir.recon(projections_normalized_images_log,
                                           angles=self.parent.final_list_of_angles_rad,
                                           num_rows = width,  
                                           num_cols = width,  # width,
                                        #    weight_type='transmission',
                                           center_offset = center_offset,
                                           max_resolutions = svmbir_parameters['max_resolutions'],
                                           sharpness = svmbir_parameters['sharpness'],
                                           snr_db = svmbir_parameters['snr_db'],
                                           positivity = svmbir_parameters['positivity'],
                                           p=1.2, 
                                           T=2.0,
                                           max_iterations = 100,
                                           num_threads = NUM_THREADS,
                                           verbose=0,
                                           roi_radius=1000,
                                           svmbir_lib_path = SVMBIR_LIB_PATH,
                                           )
            logging.info(f"\tslice: {_slice}")
            logging.info(f"\tusing rec.svmbir_reconstruction ... done!")

            self.display_reconstructed_slice(gridrec=_rec_img_gridrec, 
                                             astra=_rec_img_astra, 
                                             svmbir=_rec_img_svmbir[0], 
                                             slice=_slice)

        logging.info(f"\tdone!")
      
    def display_reconstructed_slice(self, gridrec=None, astra=None, svmbir=None, slice=None):

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        im0 = axs[0].imshow(gridrec, cmap='viridis', vmin=0)
        plt.colorbar(im0, ax=axs[0], shrink=0.5)
        axs[0].set_title(f"Slice: {slice} with Gridrec")
        axs[0].axis('off')

        im1 = axs[1].imshow(astra, cmap='viridis', vmin=0)
        plt.colorbar(im1, ax=axs[1], shrink=0.5)
        axs[1].set_title(f"Slice: {slice} ASTRA")
        axs[1].axis('off')

        if svmbir is None:
            logging.warning("SVMBIR reconstruction is None, skipping display.")
            axs[2].set_visible(False)
            plt.tight_layout()
            plt.show()
        else:
            im2 = axs[2].imshow(svmbir, cmap='viridis', vmin=0)
            plt.colorbar(im2, ax=axs[2], shrink=0.5)
            axs[2].set_title(f"Slice: {slice} SVMBIR")
            axs[2].axis('off')
            plt.tight_layout()
            plt.show()
