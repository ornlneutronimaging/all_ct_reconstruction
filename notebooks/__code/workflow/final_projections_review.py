import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from IPython.display import HTML
from ipywidgets import interactive

from __code.parent import Parent
from __code import DataType


class FinalProjectionsReview(Parent):
    
    list_runs_with_infos = None

    def run(self, array=None):

        if not list(array):
            return

        nbr_images = len(array)

        # list_angles = self.parent.final_list_of_angles
        # list_runs = self.parent.list_of_runs_to_use[DataType.sample]

        nbr_cols = 5
        nbr_rows = int(np.ceil(nbr_images / nbr_cols))

        fig, axs =  plt.subplots(nrows=nbr_rows, ncols=nbr_cols,
                                figsize=(nbr_cols*2,nbr_rows*2))
        flat_axs = axs.flatten()

        _index = 0
        # list_runs_with_infos = []
        for _row in np.arange(nbr_rows):
            for _col in np.arange(nbr_cols):
                _index = _col + _row * nbr_cols
                if _index == (nbr_images):
                    break
                # title = f"{list_runs[_index]}, {list_angles[_index]}"
                # list_runs_with_infos.append(title)
                # flat_axs[_index].set_title(title)
                im1 = flat_axs[_index].imshow(array[_index], vmin=0, vmax=1)
                plt.colorbar(im1, ax=flat_axs[_index], shrink=0.5)
           
        for _row in np.arange(nbr_rows):
            for _col in np.arange(nbr_cols):
                _index = _col + _row * nbr_cols
                flat_axs[_index].axis('off')

        # self.list_runs_with_infos = list_runs_with_infos

        plt.tight_layout()
        plt.show()

    def single_image(self, image: np.ndarray = None):

        if image is None:
            return

        fig, ax = plt.subplots()
        im1 = ax.imshow(image, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        ax.axis('off')
        plt.show()

    def stack_of_images(self, array: np.ndarray = None):
        if array is None:
            return
        
        _, width = array.shape[-2:]
        horizontal_center = width // 2

        def plot_images(index):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            image = array[index]
            ax.imshow(image, cmap='viridis', vmin=0, vmax=1)
            ax.axvline(x=horizontal_center, color='red', linestyle='--', label='Rotation Axis')
            ax.set_title(f"Image {index}")

        display_plot_images = interactive(plot_images, 
                                          index=widgets.IntSlider(min=0, 
                                                                  max=len(array)-1, 
                                                                  step=1, 
                                                                  value=0))
        display(display_plot_images)

    def list_runs_to_reject(self):
        
        label_ui = widgets.HTML("<b>Select runs you want to exclude from the final reconstruction:</b>")
        self.parent.runs_to_exclude_ui = widgets.SelectMultiple(options=self.list_runs_with_infos,
                                                                layout=widgets.Layout(height="300px"))
        display(widgets.VBox([label_ui, self.parent.runs_to_exclude_ui]))
        