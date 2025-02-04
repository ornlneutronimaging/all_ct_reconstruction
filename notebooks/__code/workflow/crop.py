import numpy as np
import logging
from neutompy.preproc.preproc import correction_COR
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
from matplotlib.patches import Rectangle

from __code.parent import Parent
from __code.utilities.configuration_file import CropRegion
from __code import crop_roi as default_roi
from __code import OperatingMode
from __code import DataType


class Crop(Parent):

    before_normalization = False

    def set_region(self, before_normalization=False):

        self.before_normalization = before_normalization
    
        width = self.parent.image_size['width']
        height = self.parent.image_size['height']

        default_left = default_roi[OperatingMode.white_beam]['left']
        default_right = default_roi[OperatingMode.white_beam]['right']
        default_top = default_roi[OperatingMode.white_beam]['top']
        default_bottom = default_roi[OperatingMode.white_beam]['bottom'] 

        if default_right < 0:
            default_right = width - abs(default_right)

        if default_bottom < 0:
            default_bottom = height - abs(default_bottom)

        if before_normalization:
            _data = self.parent.master_3d_data_array[DataType.sample]
        else:
           _data = self.parent.normalized_images
        integrated = np.min(_data, axis=0)

        max_value = np.max(integrated)

        if before_normalization:
            vmax_default_value = max_value
        else:
            vmax_default_value = 1

        def plot_crop(left, right, top, bottom, vmin, vmax):

            fig, axs = plt.subplots(figsize=(7,7))

            img = axs.imshow(integrated, vmin=vmin, vmax=vmax)
            plt.colorbar(img, ax=axs, shrink=0.5)

            width = right - left + 1
            height = bottom - top + 1

            axs.add_patch(Rectangle((left, top), width, height,
                                            edgecolor='yellow',
                                            facecolor='green',
                                            fill=True,
                                            lw=2,
                                            alpha=0.3,
                                            ),
                )     

            return left, right, top, bottom    
        
        self.display_roi = interactive(plot_crop,
                                       left=widgets.IntSlider(min=0,
                                                              max=width-1,
                                                              layout=widgets.Layout(width="100%"),
                                                              value=default_left),
                                        right=widgets.IntSlider(min=0,
                                                              layout=widgets.Layout(width="100%"),
                                                                max=width-1,
                                                                value=default_right),                      
                                        top=widgets.IntSlider(min=0,
                                                              layout=widgets.Layout(width="100%"),
                                                              max=height-1,
                                                              value=default_top),
                                        bottom=widgets.IntSlider(min=0,
                                                              layout=widgets.Layout(width="100%"),
                                                                 max=height-1,
                                                                 value=default_bottom),
                                        vmin=widgets.FloatSlider(min=0,
                                                              layout=widgets.Layout(width="100%"),
                                                                 max=max_value,
                                                                 value=0),
                                        vmax=widgets.FloatSlider(min=0,
                                                              layout=widgets.Layout(width="100%"),
                                                                 max=max_value,
                                                                 value=vmax_default_value),
                                        )
        display(self.display_roi)

    def run(self):
        left, right, top, bottom = self.display_roi.result
        self.parent.crop_region = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        self.parent.configuration.crop_region = CropRegion(left=left, right=right, top=top, bottom=bottom)
        
        if self.before_normalization:
            self.parent.master_3d_data_array[DataType.sample] = np.array([image[top: bottom+1, left: right+1] 
                                                 for image in self.parent.master_3d_data_array[DataType.sample]])
            self.parent.master_3d_data_array[DataType.ob] = np.array([image[top: bottom+1, left: right+1] 
                                                                      for image in self.parent.master_3d_data_array[DataType.ob]])
            if self.parent.master_3d_data_array[DataType.dc] is not None:
                self.parent.master_3d_data_array[DataType.dc] = np.array([image[top: bottom+1, left: right+1] 
                                                                          for image in self.parent.master_3d_data_array[DataType.dc]])

        else:
            self.parent.normalized_images = np.array([image[top: bottom+1, left: right+1] 
                                                     for image in self.parent.normalized_images])
        