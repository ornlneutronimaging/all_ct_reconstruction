import numpy as np
import logging
from neutompy.preproc.preproc import correction_COR
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
from IPython.core.display import HTML
import ipywidgets as widgets
from skimage.transform import rotate

# from imars3d.backend.diagnostics.rotation import find_rotation_center

from __code.parent import Parent
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

    manual_center_selection = None

    is_manual_mode = True

    height = None

    display_plot = None

    def settings(self):

        if self.is_manual_mode:
            value = 'Manual'
        else:
            value= 'Automatic'

        self.auto_mode_ui = widgets.RadioButtons(options=['Automatic', 'Manual'],
                                                 descriptions="Mode:",
                                                 value=value)
        display(self.auto_mode_ui)

    def is_manual_mode(self):
        try:
            return self.auto_mode_ui.value == "Manual"
        except AttributeError:
            return "Manual"

    def _isolate_0_and_180_degrees_images_white_beam_mode(self):
        logging.info(f"\tisolating 0 and 180 degres: ")
        list_of_angles = self.parent.final_list_of_angles
        self._saving_0_180_360(list_of_angles)

    def _saving_0_and_180(self, list_of_angles):
        angles_minus_180 = [float(_value) - 180 for _value in list_of_angles]
        abs_angles_minus_180 = np.abs(angles_minus_180)
        minimum_value = np.min(abs_angles_minus_180)

        index_0_degree = 0
        index_180_degree = np.where(minimum_value == abs_angles_minus_180)[0][0]
        logging.info(f"\t{index_0_degree = }")
        logging.info(f"\t{index_180_degree = }")

        # retrieve data for those indexes
        self.image_0_degree = self.parent.corrected_images[index_0_degree]
        self.image_180_degree = self.parent.corrected_images[index_180_degree]

    def _saving_0_180_360(self, list_of_angles):
        self._saving_0_and_180(list_of_angles=list_of_angles)

        angles_minus_360 = [float(_value) - 360 for _value in list_of_angles]
        abs_angles_minus_360 = np.abs(angles_minus_360)
        minimum_value = np.min(abs_angles_minus_360)

        index_360_degree = np.where(minimum_value == abs_angles_minus_360)[0][0]
        self.image_360_degree = self.parent.corrected_images[index_360_degree]
        logging.info(f"\t{index_360_degree = }")

    def _isolate_0_and_180_degrees_images(self):
        list_of_runs_to_use = self.parent.list_of_runs_to_use[DataType.sample]
        logging.info(f"\t{list_of_runs_to_use = }")
        list_of_angles = np.array([self.parent.list_of_runs[DataType.sample][_key][Run.angle] for _key in list_of_runs_to_use])
        self.parent.final_list_of_angles = list_of_angles
        self._saving_0_and_180(list_of_angles)

    def select_range(self):
        if self.parent.MODE == OperatingMode.tof:
            self._isolate_0_and_180_degrees_images()
        else:
            self._isolate_0_and_180_degrees_images_white_beam_mode()

        height = self.parent.image_size['height']

        def plot_range(y_top, y_bottom):
            _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

            axs[0].imshow(self.image_0_degree, vmin=0, vmax=1)
            axs[0].set_title("0 degree")
            axs[0].axhspan(y_top, y_bottom, color='blue', alpha=0.2)

            axs[1].imshow(self.image_180_degree, vmin=0, vmax=1)
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

    def run(self):
        if self.auto_mode_ui.value == 'Automatic':
            self.calculate_using_neutompy()
        else:
            self.using_manual_mode()

    def using_manual_mode(self):
        self.manual_center_of_rotation()
        # self.manual_tilt_correction()

    def get_center_of_rotation(self):
        return self.manual_center_selection.result

    def manual_center_of_rotation(self):
        display(HTML("Center of rotation"))

        width = self.parent.image_size['width']
        vmax = np.max([self.image_0_degree, self.image_180_degree, self.image_360_degree])

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
                                                                      max=width-1, 
                                                                      layout=widgets.Layout(width="100%"),
                                                                      value=int(width/2)),
                                    v_range = widgets.FloatRangeSlider(min=0,
                                                                       max=vmax,
                                                                       layout=widgets.Layout(width='100%'),
                                                                       value=[0, vmax]),

                                    )                                                                     
        display(self.manual_center_selection)

    # def manual_tilt_correction(self):
    #     display(HTML("Tilt correction"))

    #     width = self.parent.image_size['width']
    #     vmax = np.max([self.image_0_degree, self.image_180_degree, self.image_360_degree])

    #     def plot_images(angles, tilt, verti_guide, guide_width, v_range):
    #         _, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    #         at_least_one_image_selected = False
    #         list_images = []
    #         if ImageAngles.degree_0 in angles:
    #             list_images.append(self.image_0_degree)
    #             at_least_one_image_selected = True
    #         if ImageAngles.degree_180 in angles:
    #             list_images.append(self.image_180_degree)
    #             at_least_one_image_selected = True
    #         if ImageAngles.degree_360 in angles:
    #             list_images.append(self.image_360_degree)
    #             at_least_one_image_selected = True

    #         if not at_least_one_image_selected:
    #             return

    #         if len(list_images) > 1:
    #             final_image = np.mean(np.array(list_images), axis=0)
    #         else:
    #             final_image = list_images[0]

    #         rotated_image = 

    #         axs.imshow(final_image, vmin=v_range[0], vmax=v_range[1])
    #         axs.axvline(verti_guide, color='blue', linestyle='--')
    #         axs.axvline(verti_guide-int(guide_width/2), color='yellow')
    #         axs.axvline(verti_guide+int(guide_width/2), color='yellow')

    #     display_plot = interactive(plot_images,
    #                                angles=widgets.SelectMultiple(options=[ImageAngles.degree_0, ImageAngles.degree_180, ImageAngles.degree_360],
    #                                                                   value=[ImageAngles.degree_0, ImageAngles.degree_180]),
    #                                tilt=widgets.FloatSlider(min=-5, max=5, value=0, step=0.01),
    #                                verti_guide=widgets.IntSlider(min=0, 
    #                                                                   max=width-1, 
    #                                                                   value=int(width/2)),
    #                                 guide_width = widgets.IntSlider(min=0,
    #                                                                max=width,
    #                                                                value=30),
    #                                 v_range = widgets.FloatRangeSlider(min=0,
    #                                                                    max=vmax,
    #                                                                    value=[0, vmax]),
    #                                 )                                                                     
    #     display(display_plot)


    def calculate_using_neutompy(self):
        
        # retrieve index of 0 and 180degrees runs
        logging.info(f"calculate center of rotation:")

        logging_3d_array_infos(message="before", array=self.parent.corrected_images)

        corrected_images = np.array(self.parent.corrected_images) if type(self.parent.corrected_images) == list else self.parent.corrected_images

        y_top, y_bottom = self.display_plot.result

        # update configuration
        self.parent.configuration.range_of_slices_for_center_of_rotation = list([y_top, y_bottom])

        mid_point = int(np.mean([y_top, y_bottom]))
        rois = ((y_top, mid_point+1), (mid_point, y_bottom))

        corrected_images = correction_COR(corrected_images,
                       np.array(self.image_0_degree),
                       np.array(self.image_180_degree),
                       rois=rois)
        logging.info(f"{np.shape(corrected_images) =}")
        self.parent.corrected_images = corrected_images

        logging_3d_array_infos(message="after", array=self.parent.corrected_images)

        # update configuration
        self.parent.configuration.calculate_center_of_rotation = True
