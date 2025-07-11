import os
import logging
import glob
from IPython.display import display
import ipywidgets as widgets
from IPython.display import HTML
import matplotlib.pyplot as plt
from ipywidgets import interactive
import numpy as np

from __code import OperatingMode, DataType
from __code.utilities.logging import setup_logging
from __code.utilities.file_folder_browser import FileFolderBrowser
from __code.utilities.load import load_data_using_multithreading
from __code.config import PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION, DEBUG, debug_folder

LOG_BASENAME_FILENAME, _ = os.path.splitext(os.path.basename(__file__))


class Step4VisualizationCcdImages:
   
    configuration = None

    primary_reconstructed_slices = None
    secondary_reconstructed_slices = None

    primary_list_tiff = None
    secondary_list_tiff = None

    def __init__(self, system=None):
        # self.configuration = Configuration()
        self.working_dir = system.System.get_working_dir()
        if DEBUG:
            self.working_dir = debug_folder[OperatingMode.white_beam][DataType.extra]

        self.instrument = system.System.get_instrument_selected()

        setup_logging(LOG_BASENAME_FILENAME)      
        logging.info(f"working_dir: {self.working_dir}")
        logging.info(f"instrument: {self.instrument}")
        if DEBUG:
            logging.info(f"WARNING!!!! we are running using DEBUG mode!")

    def select_reconstructed_folder(self, primary_reconstructed=True):
        self.primary_reconstructed = primary_reconstructed

        working_dir = self.working_dir
        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.save_folder_selected)
        o_file_browser.select_input_folder(instruction=f"Select folder containing the reconstructed images ...",
                                           multiple_flag=False)
        
    def save_folder_selected(self, folder_selected):
        if self.primary_reconstructed:
            self.primary_reconstructed_folder = folder_selected
        else:
            self.secondary_reconstructef_folder = folder_selected

        logging.info(f"folder_selected: {folder_selected} (primary_reconstructed={self.primary_reconstructed})")
        list_tiff = glob.glob(os.path.join(folder_selected, "*.tif*"))
        logging.info(f"found {len(list_tiff)} tiff files in {folder_selected}")

        if self.primary_reconstructed:
            self.primary_list_tiff = list_tiff
        else:
            self.secondary_list_tiff = list_tiff

    def load_images(self):
        if self.primary_reconstructed:
            list_tiff = self.primary_list_tiff
        else:
            list_tiff = self.secondary_list_tiff
              
        nbr_images_to_use = int(self.percentage_to_use.value / 100 * len(list_tiff))
        list_tiff_index_to_use = np.random.randint(0, len(list_tiff), nbr_images_to_use)
        list_tiff_index_to_use.sort()
        list_tiff = [list_tiff[_index] for _index in list_tiff_index_to_use]

        logging.info(f"loading {len(list_tiff)} images ...")
        
        if self.primary_reconstructed:
            self.primary_reconstructed_slices = load_data_using_multithreading(list_tiff)
        else:
            self.secondary_reconstructed_slices = load_data_using_multithreading(list_tiff)
        
        logging.info(f"done!")
        print(f"Loading done! ({len(list_tiff)} images loaded)")
        
    def select_percentage_of_images_to_visualize(self):
        self.percentage_to_use = widgets.FloatSlider(value=PERCENTAGE_OF_DATA_TO_USE_FOR_RECONSTRUCTION,
                                    min=.1,
                                    max=100,
                                    step=.1,
                                    layout=widgets.Layout(width='100%'))
        display(self.percentage_to_use)

        list_of_tiff = self.primary_list_tiff
        percentage = self.percentage_to_use.value
        nbr_images = int(percentage / 100 * len(list_of_tiff))
        self.number_of_images_to_use = widgets.Label(f"{nbr_images} primary slices will be used for the visualization")
        display(self.number_of_images_to_use)
        self.percentage_to_use.observe(self.on_percentage_to_use_change, names='value') 

    def on_percentage_to_use_change(self, change):
        new_value = change['new']
        list_tiff = self.primary_list_tiff
        nbr_images = int(new_value / 100 * len(list_tiff))
        self.number_of_images_to_use.value = f"{nbr_images} images will be used for the visualization"

    def visualize(self):

        if self.primary_reconstructed_slices is None:
            logging.info(f"Nothing to visualize!")
            print("Nothing to visualize (load the primary reconstructed stack of images!")
            return
        
        disable_secondary_slider = False

        if self.secondary_reconstructed_slices is None:
            disable_secondary_slider = True
            len_secondary_slices = 1
        else:
            len_secondary_slices = len(self.secondary_reconstructed_slices)

        def plot_norm(left_index=0, right_index=0):
            
            if self.secondary_reconstructed_slices is not None:

                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                im0 = axs[0].imshow(self.primary_reconstructed_slices[left_index], cmap='viridis')
                axs[0].set_title(f"Primary reconstructed slice #{left_index}")
                plt.colorbar(im0, ax=axs[0], shrink=0.5)
                                
                im1 = axs[1].imshow(self.secondary_reconstructed_slices[right_index], cmap='viridis')
                axs[1].set_title(f"Secondary reconstructed slice #{right_index}")
                plt.colorbar(im1, ax=axs[1], shrink=0.5)
              
                plt.tight_layout()
                plt.show()

            else:

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
                im0 = axs.imshow(self.primary_reconstructed_slices[left_index], cmap='viridis')
                axs.set_title(f"Primary reconstructed slice #{left_index}")
                plt.colorbar(im0, ax=axs, shrink=0.5)

        interactive_plot = interactive(plot_norm,
                                       left_index=widgets.IntSlider(min=0,
                                                                      max=len(self.primary_reconstructed_slices)-1,
                                                                      step=1,
                                                                      value=0),
                                        right_index=widgets.IntSlider(min=0,
                                                                      max=len_secondary_slices-1,
                                                                      step=1,
                                                                      value=0,
                                                                      disabled=disable_secondary_slider),
                                        )                            
        
        display(interactive_plot)
        
        logging.info(f"Visualization of {len(self.primary_reconstructed_slices)} primary slices done!")
        if self.secondary_reconstructed_slices is not None:
            logging.info(f"Visualization of {len(self.secondary_reconstructed_slices)} secondary slices done!")
        logging.info(f"Done!")
                