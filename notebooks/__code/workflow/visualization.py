from IPython.display import display
import ipywidgets as widgets
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from ipywidgets import interactive
import numpy as np

from __code.parent import Parent
from __code import DataType
from __code.workflow.final_projections_review import FinalProjectionsReview


class Visualization(Parent):

    def settings(self):
        self.display_ui = widgets.ToggleButtons(options=['1 image at a time',
                                                         'All images'],
                                                         description="How to plot?",
                                                         )
        display(self.display_ui)

    def visualize(self, data_before=None, data_after=None, label_before="", label_after="", turn_on_vrange=False):

        if self.display_ui.value == '1 image at a time':

            list_of_images = self.parent.list_of_images[DataType.sample]
            
            if turn_on_vrange:

                vmin_before = np.min(data_before)
                vmax_before = np.max(data_before)
                vmin_after = np.min(data_after)
                vmax_after = np.max(data_after)

                def plot_norm(image_index=0, 
                              vmin_before=vmin_before, vmax_before=vmax_before, 
                              vmin_after=vmin_after, vmax_after=vmax_after):

                    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

                    _norm_data = data_after[image_index]
                    # _run_number = list_of_images[image_index]
                    _raw_data = data_before[image_index]

                    im0 = axs[0].imshow(_raw_data, vmin=vmin_before, vmax=vmax_before)
                    axs[0].set_title(label_before)
                    plt.colorbar(im0, ax=axs[0], shrink=0.5)

                    im1 = axs[1].imshow(_norm_data, vmin=vmin_after, vmax=vmax_after)
                    axs[1].set_title(label_after)
                    plt.colorbar(im1, ax=axs[1], shrink=0.5)
            
                    # fig.set_title(f"{_run_number}")
                    
                    plt.tight_layout()
                    plt.show()

                display_plot = interactive(plot_norm,
                                        image_index=widgets.IntSlider(min=0,
                                                                        max=len(list_of_images)-1,
                                                                        value=0),
                                        vmin_before=widgets.IntSlider(min=vmin_before, max=vmax_before, value=vmin_before),
                                        vmax_before=widgets.IntSlider(min=vmin_before, max=vmax_before, value=vmax_before),
                                        vmin_after=widgets.FloatSlider(min=vmin_after, max=vmax_after, value=0),
                                        vmax_after=widgets.FloatSlider(min=vmin_after, max=vmax_after, value=1))

            else:

                def plot_norm(image_index=0):

                    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

                    _norm_data = data_after[image_index]
                    # _run_number = list_of_images[image_index]
                    _raw_data = data_before[image_index]

                    im0 = axs[0].imshow(_raw_data)
                    axs[0].set_title(label_before)
                    plt.colorbar(im0, ax=axs[0], shrink=0.5)

                    im1 = axs[1].imshow(_norm_data)
                    axs[1].set_title(label_after)
                    plt.colorbar(im1, ax=axs[1], shrink=0.5)
            
                    # fig.set_title(f"{_run_number}")
                    
                    plt.tight_layout()
                    plt.show()

                display_plot = interactive(plot_norm,
                                        image_index=widgets.IntSlider(min=0,
                                                                        max=len(list_of_images)-1,
                                                                        value=0),
                )
                
            display(display_plot)


        else:
            o_review = FinalProjectionsReview(parent=self.parent)
            o_review.run(array=data_after)

    def visualize_all_images_at_once(self):
        
        master_3d_data_array = self.parent.master_3d_data_array

        nbr_cols = 8

        for _data_type in master_3d_data_array.keys():

            if not self.parent.list_of_images[_data_type]:
                continue
            
            display(HTML(f"<b>{_data_type}</b>"))
            array = master_3d_data_array[_data_type]
            nbr_images = len(self.parent.list_of_images[_data_type])
            nbr_rows = int(np.ceil(nbr_images / nbr_cols))

            fig, axs =  plt.subplots(nrows=nbr_rows, ncols=nbr_cols,
                                    figsize=(nbr_cols*2,nbr_rows*2))
            flat_axs = axs.flatten()

            _index = 0
            list_runs_with_infos = []
            for _row in np.arange(nbr_rows):
                for _col in np.arange(nbr_cols):
                    _index = _col + _row * nbr_cols
                    if _index == (nbr_images):
                        break
                    # title = f"{list_runs[_index]}, {list_angles[_index]}"
                    # list_runs_with_infos.append(title)
                    # flat_axs[_index].set_title(title)
                    im1 = flat_axs[_index].imshow(array[_index])
                    plt.colorbar(im1, ax=flat_axs[_index], shrink=0.5)
            
            for _row in np.arange(nbr_rows):
                for _col in np.arange(nbr_cols):
                    _index = _col + _row * nbr_cols
                    flat_axs[_index].axis('off')

            plt.tight_layout()
            plt.show()


