import numpy as np
import logging
from neutompy.preproc.preproc import correction_COR, find_COR
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
from IPython.core.display import HTML
import ipywidgets as widgets
from skimage.transform import rotate
from algotom.prep.calculation import find_center_vo, find_center_360

# from imars3d.backend.diagnostics.rotation import find_rotation_center

from __code.parent import Parent
from __code.config import NUM_THREADS
from __code import DataType, Run, OperatingMode
from __code.utilities.logging import logging_3d_array_infos


class ImageAngles:
    
    degree_0 = '0 degree'
    degree_180 = '180 degree'
    degree_360 = '360 degree'


class CenterOfRotationAndTilt(Parent):

    image_0_degree = None
    image_180_degree = None
    image_360_degree = None

    index_0_degree = 0
    index_180_degree = 0
    index_360_degree = 0

    manual_center_selection = None

    is_manual_mode = False

    height = None

    display_plot = None

    def _isolate_0_and_180_degrees_images_white_beam_mode(self):
        logging.info(f"\tisolating 0 and 180 degres: ")
        list_of_angles = self.parent.final_list_of_angles
        self._saving_0_and_180(list_of_angles)

    def _saving_0_and_180(self, list_of_angles):
        angles_minus_180 = [float(_value) - 180 for _value in list_of_angles]
        abs_angles_minus_180 = np.abs(angles_minus_180)
        minimum_value = np.min(abs_angles_minus_180)

        index_0_degree = 0
        index_180_degree = np.where(minimum_value == abs_angles_minus_180)[0][0]
        self.index_180_degree = index_180_degree

        logging.info(f"\t{index_0_degree = }")
        logging.info(f"\t{index_180_degree = }")

        # retrieve data for those indexes
        self.image_0_degree = self.parent.normalized_images_log[index_0_degree]
        self.image_180_degree = self.parent.normalized_images_log[index_180_degree]

    def _saving_360(self, list_of_angles):
        angles_minus_360 = [float(_value) - 360 for _value in list_of_angles]
        abs_angles_minus_360 = np.abs(angles_minus_360)
        minimum_value = np.min(abs_angles_minus_360)

        index_360_degree = np.where(minimum_value == abs_angles_minus_360)[0][0]
        self.index_360_degree = index_360_degree
        self.image_360_degree = self.parent.normalized_images_log[index_360_degree]
        logging.info(f"\t{index_360_degree = }")

    def isolate_0_180_360_degrees_images(self):
        list_of_angles = self.parent.final_list_of_angles
        self._saving_0_and_180(list_of_angles)
        self._saving_360(list_of_angles)

    def _isolate_0_and_180_degrees_images(self):
        list_of_angles = self.parent.final_list_of_angles = list_of_angles
        self._saving_0_and_180(list_of_angles)

    def select_range(self):
        if self.parent.MODE == OperatingMode.tof:
            self.isolate_0_180_360_degrees_images()
        else:
            self._isolate_0_and_180_degrees_images_white_beam_mode()

        height, _ = np.shape(self.image_0_degree)

        def plot_range(y_top, y_bottom):
            _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

           # im0 = axs[0].imshow(self.image_0_degree, vmin=0, vmax=1)
            im0 = axs[0].imshow(self.image_0_degree)
            plt.colorbar(im0, ax=axs[0])
            axs[0].set_title("0 degree")
            axs[0].axhspan(y_top, y_bottom, color='blue', alpha=0.2)

            #im1 = axs[1].imshow(self.image_180_degree, vmin=0, vmax=1)
            im1 = axs[1].imshow(self.image_180_degree)
            plt.colorbar(im1, ax=axs[1])
            axs[1].set_title("180 degree")
            axs[1].axhspan(y_top, y_bottom, color='blue', alpha=0.2)

            plt.tight_layout()
            plt.show()

            return y_top, y_bottom

        self.display_plot = interactive(plot_range,
                                   y_top = widgets.IntSlider(min=0, 
                                                            max=height-1, 
                                                            value=0),
                                   y_bottom = widgets.IntSlider(min=0,
                                                            max=height-1, 
                                                            value=height-1),
        )

        display(self.display_plot)

    # ---- tilt correction ----
    def run_tilt_correction(self):
       # self.calculate_tilt_using_neutompy()
        self.calculate_and_apply_tilt_using_neutompy()

    def calculate_tilt_using_neutompy(self):
        logging.info(f"calculate tilt correction:")
        proj_crop_min = np.min(self.parent.normalized_images_log, axis=0)
        pixel_offset, self.tilt_angle = find_COR(self.image_0_degree, 
                                            self.image_180_degree,
                                            nroi=1,
                                            ref_proj=proj_crop_min)

        print(f"\t{pixel_offset = }")
        print(f"\t{self.tilt_angle = }")

    def calculate_and_apply_tilt_using_neutompy(self):
        
        # retrieve index of 0 and 180degrees runs
        logging.info(f"calculate and apply tilt correction:")

        logging_3d_array_infos(message="before tilt correction", array=self.parent.normalized_images_log)

        normalized_images = np.array(self.parent.normalized_images_log) if type(self.parent.normalized_images_log) == list else self.parent.normalized_images_log

        y_top, y_bottom = self.display_plot.result

        # update configuration
        self.parent.configuration.range_of_slices_for_center_of_rotation = list([y_top, y_bottom])

        mid_point = int(np.mean([y_top, y_bottom]))
        rois = ((y_top, mid_point+1), (mid_point, y_bottom))

        logging.info(f"\t{np.shape(normalized_images) =}")
        logging.info(f"\t{np.shape(self.image_0_degree) =}")
        logging.info(f"\t{np.shape(self.image_180_degree) =}")
        logging.info(f"\t{rois =}")

        normalized_images = correction_COR(normalized_images,
                       np.array(self.image_0_degree),
                       np.array(self.image_180_degree),
                       shift=None,
                       theta=None,
                       rois=rois)
        logging.info(f"{np.shape(normalized_images) =}")
        self.parent.normalized_images_log = normalized_images

        logging_3d_array_infos(message="after tilt correction", array=self.parent.normalized_images_log)

    #  ---- center of rotation -----
    def center_of_rotation_settings(self):
        if self.is_manual_mode:
            value = 'Manual'
        else:
            value= 'Automatic'

        self.auto_mode_ui = widgets.RadioButtons(options=['Automatic', 'Manual'],
                                                 descriptions="Mode:",
                                                 value=value)
        display(self.auto_mode_ui)
    
    # def is_manual_mode(self):
    #     try:
    #         return self.auto_mode_ui.value == "Manual"
    #     except AttributeError:
    #         return "Manual"

    def using_manual_mode(self):
        self.manual_center_of_rotation()
        # self.manual_tilt_correction()

    def get_center_of_rotation(self):
        return self.manual_center_selection.result

    def manual_center_of_rotation(self):
        display(HTML("Center of rotation"))

        _, width = np.shape(self.image_0_degree)
        vmax = np.max([self.image_0_degree, self.image_180_degree, self.image_360_degree])
        vmax = 4 # debug

        def plot_images(angles, center, v_range):
            _, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

            at_least_one_image_selected = False
            list_images = []
            if ImageAngles.degree_0 in angles:
                list_images.append(self.image_0_degree)
                at_least_one_image_selected = True
            if ImageAngles.degree_180 in angles:
                list_images.append(self.image_180_degree)
                at_least_one_image_selected = True
            if ImageAngles.degree_360 in angles:
                list_images.append(self.image_360_degree)
                at_least_one_image_selected = True

            if not at_least_one_image_selected:
                return

            if len(list_images) > 1:
                final_image = np.mean(np.array(list_images), axis=0)
            else:
                final_image = list_images[0]

            axs.imshow(final_image, vmin=v_range[0], vmax=v_range[1], cmap='viridis')
            axs.axvline(center, color='blue', linestyle='--')

            return center

        self.manual_center_selection = interactive(plot_images,
                                   angles=widgets.SelectMultiple(options=[ImageAngles.degree_0, ImageAngles.degree_180, ImageAngles.degree_360],
                                                                      value=[ImageAngles.degree_0, ImageAngles.degree_180]),
                                   center=widgets.IntSlider(min=0, 
                                                                      max=int(width-1), 
                                                                      layout=widgets.Layout(width="100%"),
                                                                      value=int(width/2)),
                                    v_range = widgets.FloatRangeSlider(min=0,
                                                                       max=vmax,
                                                                       layout=widgets.Layout(width='100%'),
                                                                       value=[0, vmax]),

                                    )                                                                     
        display(self.manual_center_selection)

    def run_center_of_rotation(self):
        if self.auto_mode_ui.value == "Manual":
            self.manual_center_of_rotation()
        else:
            self.select_180_or_360_degree_mode()

        # update configuration
        self.parent.configuration.calculate_center_of_rotation = True

    def select_180_or_360_degree_mode(self):
        logging.info(f"select 180 or 360 degree mode to calculate center of rotation")

        image_0_degree = self.image_0_degree
        image_180_degree = self.image_180_degree
        # image_360_degree = self.image_360_degree

        # logging.info(f"{image_0_degree.dtype = }")
        # logging.info(f"{np.shape(image_0_degree) = }")
        # logging.info(f"{image_180_degree.dtype = }")
        # logging.info(f"{np.shape(image_180_degree) = }")
        # logging.info(f"{image_360_degree.dtype = }")
        # logging.info(f"{np.shape(image_360_degree) = }")

        height, _ = np.shape(image_0_degree)
        self.slide_value = int(height/2)

        display(widgets.HTML("Select the slice to use to calculate the center of rotation"))
        # max_value = np.max([image_0_degree, image_180_degree, image_360_degree])
        max_value = np.max([image_0_degree, image_180_degree])
        # max_value = 4 # DEBUG

        def plot_images(slice_value=int(height/2), vmin=0, vmax=max_value):

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
            
            axs[0].imshow(image_0_degree, cmap='viridis', vmin=vmin, vmax=vmax)
            axs[0].set_title("0 / 0")
            axs[0].axhline(slice_value, color='blue', linestyle='--')

            axs[1].imshow(image_180_degree, cmap='viridis', vmin=vmin, vmax=vmax)
            axs[1].set_title(f"{self.parent.final_list_of_angles[self.index_180_degree]} / 180")
            axs[1].axhline(slice_value, color='blue', linestyle='--')

            # axs[2].imshow(image_360_degree, cmap='viridis', vmin=vmin, vmax=vmax)
            # axs[2].set_title(f"{self.parent.final_list_of_angles[self.index_360_degree]} / 360")
            # axs[2].axhline(slice_value, color='blue', linestyle='--')

            plt.tight_layout()

            return slice_value

        display(widgets.HTML("Measured / Expected (angles in degrees)"))

        self.plot_slice_to_use = interactive(plot_images,
                                             slice_value = widgets.IntSlider(min=0, 
                                                                             max=height-1, 
                                                                             value=int(height/2),
                                                                             layout=widgets.Layout(width="75%")),
                                             vmin=widgets.FloatSlider(min=0, 
                                                                      max=max_value, 
                                                                      value=0,
                                                                      layout=widgets.Layout(width="75%")),
                                             vmax=widgets.FloatSlider(min=0, 
                                                                      max=max_value, 
                                                                      value=max_value,
                                                                      layout=widgets.Layout(width="75%"),
                                                                      ),
        )
        display(self.plot_slice_to_use)

        display(widgets.HTML("Horizontal line shows the slide used to calculate the center of rotation"))
        display(widgets.HTML("<hr>"))

        # display(widgets.HTML("Do you want to use all the data from 0 to 180, or 0 to 360 degrees to calculate the center of rotation?"))
        # self.cor_selection = widgets.RadioButtons(options=["180 degree", "360 degree"],
        #                                  descriptions="Select mode:")
        # display(self.cor_selection)

    def calculate_center_of_rotation(self):
        self.calc_cor_with_algotom()

    def calc_cor_with_algotom(self):

        if self.auto_mode_ui.value == "Manual":
            print(f"center of rotation selected: {self.manual_center_selection.result}")
            self.parent.configuration.center_of_rotation = self.manual_center_selection.result
            return
        
        logging.info(f"calculate center of rotation using algotom (auto mode)")
       
        logging.info(f"\tworking with sinogram in log mode!")
        sinogram_of_normalized_images_log = np.moveaxis(self.parent.normalized_images_log, 1, 0) # [slice, angle, width]
        logging.info(f"{np.shape(sinogram_of_normalized_images_log) = }")

        slice_value = self.plot_slice_to_use.result

        # if self.cor_selection.value == "180 degree":
        center_of_rotation = find_center_vo(sinogram_of_normalized_images_log[self.slide_value][0:slice_value],
                                            ncore=NUM_THREADS)
        # else:
        #     try:
        #         center_of_rotation = find_center_360(sinogram_of_normalized_images_log[self.slide_value][:],
        #                                             win_width=800,
        #                                             ncore=NUM_THREADS)
        #         if type(center_of_rotation) == list:
        #             center_of_rotation = center_of_rotation[0]

        #     except ValueError as e:
        #         logging.error(f"Error: {e}")
        #         logging.info(f"Error: {e}")
        #         center_of_rotation = None
       
        logging.info(f"center of rotation = {center_of_rotation}")
        self.parent.configuration.center_of_rotation = center_of_rotation
    
    def test_center_of_rotation_calculated(self):
        center_of_rotation_calculated = self.parent.configuration.center_of_rotation

        image_0_degree = self.image_0_degree
        image_180_degree = self.image_180_degree

        combined_images = 0.5*image_0_degree + 0.5*image_180_degree
        vmax = np.max(combined_images)
        vmin = 0

        def plot_result(v_range):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            
            im = ax.imshow(combined_images, cmap='viridis', vmin=v_range[0], vmax=v_range[1])
            plt.colorbar(im, ax=ax, shrink=0.5)
            ax.axvline(center_of_rotation_calculated, color='blue', linestyle='--')

            plt.tight_layout()

        manual_center_selection = interactive(plot_result,
                                    v_range = widgets.FloatRangeSlider(min=vmin,
                                                                       max=vmax,
                                                                       layout=widgets.Layout(width='100%'),
                                                                       value=[0, vmax]),

                                    )                                                                     
        display(manual_center_selection)
