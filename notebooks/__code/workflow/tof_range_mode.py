import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display, HTML
import ipywidgets as widgets
import logging
from matplotlib.patches import Rectangle
import numpy as np

from neutronbraggedge.experiment_handler.tof import TOF
from neutronbraggedge.experiment_handler.experiment import Experiment

from __code.parent import Parent
from __code import LAMBDA, ANGSTROMS, NBR_TOF_RANGES
from __code.config import DISTANCE_SOURCE_DETECTOR


class TofRangeMode(Parent):
    
    tof_array_s = None
    lambda_array_angstroms = None

    def run(self):
        
        self.retrieve_tof_array()
        self.display_widgets()
        # self.calculate_lambda_array()
        self.plot()

    def retrieve_tof_array(self):
        logging.info(f"retrieving tof array:")
        spectra_file_full_path = self.parent.spectra_file_full_path
        logging.info(f"\t{spectra_file_full_path = }")
        _tof_handler = TOF(filename=spectra_file_full_path)
        self.tof_array_s = _tof_handler.tof_array
        logging.info(f"\ttof_array_s: {self.tof_array_s}")

    def calculate_lambda_array(self, detector_offset=9000):
        _exp = Experiment(tof=self.tof_array_s,
            distance_source_detector_m=DISTANCE_SOURCE_DETECTOR,
            detector_offset_micros=detector_offset,
                )
        _lambda_array = _exp.lambda_array
        self.lambda_array_angstroms = _lambda_array * 1e10

    def display_widgets(self):
        distance_uis = widgets.HBox([widgets.Label("distance source detector (m):",
                                               layout=widgets.Layout(width="max-contents")),
                                widgets.Label(f"{DISTANCE_SOURCE_DETECTOR}"),
        ])
        display(distance_uis)

    #     detector_offset_uis = widgets.HBox([widgets.Label("Detector offset (micros)",
    #                                            layout=widgets.Layout(width="max-contents")),
    #                                         widgets.FloatText(value=9000,
    #                                                           )])
    #     display(widgets.VBox([distance_uis, detector_offset_uis]))
    #     self.detector_offset_text_ui = detector_offset_uis.children[1]
    #     self.detector_offset_text_ui.observe(self.detector_value_changed, names='value')


    # def detector_value_changed(self, value):
    #     detector_offset = value['new']
    #     self.calculate_lambda_array(detector_offset=detector_offset)
    #     self.plot()

    def plot(self):
        
        data_3d = self.parent.data_3d_of_all_projections_merged
        integrated_data_3d = np.sum(data_3d, axis=0)

        width = self.parent.image_size['width']
        height = self.parent.image_size['height']

        def display_projection_and_profile(det_offset=9000, left=100, top=100, width=100, height=100, 
                            x_move=0, y_move=0, y_log_scale=True):

            self.calculate_lambda_array(detector_offset=det_offset)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            im0 = axs[0].imshow(integrated_data_3d)
            # plt.colorbar(im0, ax=axs[0], shrink=0.8)

            new_left = left + x_move
            new_top = top + y_move

            axs[0].add_patch(Rectangle((new_left, new_top), width, height,
                                        edgecolor='yellow',
                                        facecolor='green',
                                        fill=True,
                                        lw=2,
                                        alpha=0.3,
                                        ),
            )

            counts_of_region = np.mean(np.mean(data_3d[:, new_top: new_top+height+1, 
                                                       new_left: new_left+width+1], 
                                                       axis=1), axis=1)
            self.y_axis = counts_of_region
            self.y_log_scale = y_log_scale

            if y_log_scale:
                axs[1].semilogy(self.lambda_array_angstroms, counts_of_region)
            else:
                axs[1].plot(self.lambda_array_angstroms, counts_of_region)           
            axs[1].set_xlabel(f"{LAMBDA} ({ANGSTROMS})")
            axs[1].set_ylabel(f"Mean counts of region selected")

            plt.tight_layout()
            plt.show()

        self.plot_profile = interactive(display_projection_and_profile,
                                        det_offset=widgets.FloatText(value=9000),
                                        left=widgets.IntSlider(min=0, max=width-1, value=100),
                                        top=widgets.IntSlider(min=0, max=height-1, value=100),
                                        width=widgets.IntSlider(min=0, max=width, value=100),
                                        height=widgets.IntSlider(min=0, max=height, value=100),
                                        x_move=widgets.IntSlider(min=-width, max=width, value=0),
                                        y_move=widgets.IntSlider(min=-height, max=height, value=0),
                                        y_log_scale=widgets.Checkbox(value=True),
                                        )
        display(self.plot_profile)

    def select_tof_range(self):

            max_y = np.max(self.y_axis)

            def display_profile(left_tof, right_tof):

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

                x_axis = self.lambda_array_angstroms
                y_axis = self.y_axis

                if self.y_log_scale:
                    axs.semilogy(x_axis, y_axis)
                else:
                    axs.plot(x_axis, y_axis)

                axs.annotate(f"{self.lambda_array_angstroms[left_tof]:0.3f} {ANGSTROMS}",
                            (self.lambda_array_angstroms[left_tof], max_y),
                            color='red',
                            horizontalalignment='right',
                            )

                axs.annotate(f"{self.lambda_array_angstroms[right_tof]:0.3f} {ANGSTROMS}",
                            (self.lambda_array_angstroms[right_tof], max_y),
                            color='red',
                            horizontalalignment='left',
                            )


                axs.axvspan(self.lambda_array_angstroms[left_tof], 
                            self.lambda_array_angstroms[right_tof], 
                            alpha=0.5,
                            linestyle="--",
                            edgecolor='green')
                plt.show()

                return left_tof, right_tof

            self.plot_tof_profile = interactive(display_profile,
                                                left_tof=widgets.IntSlider(min=0,
                                                                            max=len(self.lambda_array_angstroms)-1,
                                                                            value=0),
                                                right_tof=widgets.IntSlider(min=0,
                                                                            max=len(self.lambda_array_angstroms)-1,
                                                                            value=len(self.lambda_array_angstroms)-1),
            )
            display(self.plot_tof_profile)

    def combine_tof_mode_data(self):
        
        logging.info(f"combining in tof mode:")
        left_tof_index, right_tof_index = self.plot_tof_profile.result
        logging.info(f"\tfrom index {left_tof_index} ({self.lambda_array_angstroms[left_tof_index]:0.3f} {ANGSTROMS})")
        logging.info(f"\tto index {right_tof_index} ({self.lambda_array_angstroms[right_tof_index]:0.3f} {ANGSTROMS})")
        self.parent.configuration.range_of_tof_to_combine = [(left_tof_index, right_tof_index)]

        master_3d_data_array = self.parent.master_3d_data_array

        for _data_type in master_3d_data_array.keys():
            logging.info(f"\tbefore: {np.shape(master_3d_data_array[_data_type]) = }")
            new_master_3d_data_array = []
            for _data in master_3d_data_array[_data_type]:
                new_master_3d_data_array.append(np.mean(_data[left_tof_index:right_tof_index+1, :, :], axis=0))
            master_3d_data_array[_data_type] = np.array(new_master_3d_data_array)
            logging.info(f"\tafter: {np.shape(master_3d_data_array[_data_type]) = }")
        self.parent.master_3d_data_array = master_3d_data_array


    # def select_multi_tof_range(self):

    #     self.list_of_tof_ranges_ui = []
    #     list_default_use_it = [False for i in range(NBR_TOF_RANGES)]
    #     list_default_use_it[0] = True
    #     max_y = np.max(self.y_axis)

    #     for i in range(NBR_TOF_RANGES):

    #         def display_profile(use_it, left_tof, right_tof):

    #             fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    #             axs.set_title(f"TOF range {i}/{NBR_TOF_RANGES-1}")

    #             x_axis = self.lambda_array_angstroms
    #             y_axis = self.y_axis

    #             if self.y_log_scale:
    #                 axs.semilogy(x_axis, y_axis)
    #             else:
    #                 axs.plot(x_axis, y_axis)

    #             axs.annotate(f"{self.lambda_array_angstroms[left_tof]:0.3f} {ANGSTROMS}",
    #                         (self.lambda_array_angstroms[left_tof], max_y),
    #                         color='red',
    #                         horizontalalignment='right',
    #                         )

    #             axs.annotate(f"{self.lambda_array_angstroms[right_tof]:0.3f} {ANGSTROMS}",
    #                         (self.lambda_array_angstroms[right_tof], max_y),
    #                         color='red',
    #                         horizontalalignment='left',
    #                         )


    #             axs.axvspan(self.lambda_array_angstroms[left_tof], 
    #                         self.lambda_array_angstroms[right_tof], 
    #                         alpha=0.5,
    #                         linestyle="--",
    #                         edgecolor='green')
    #             plt.show()

    #             return use_it, left_tof, right_tof

    #         _plot_tof_profile = interactive(display_profile,
    #                                             use_it=widgets.Checkbox(value=list_default_use_it[i]),
    #                                             left_tof=widgets.IntSlider(min=0,
    #                                                                         max=len(self.lambda_array_angstroms)-1,
    #                                                                         value=0),
    #                                             right_tof=widgets.IntSlider(min=0,
    #                                                                         max=len(self.lambda_array_angstroms)-1,
    #                                                                         value=len(self.lambda_array_angstroms)-1),
    #         )
    #         display(_plot_tof_profile)
    #         display(HTML("<hr style='border-bottom: dotted 1px;'>"))

    #     self.list_of_tof_ranges_ui.append(_plot_tof_profile)
 
    # def combine_tof_mode_data(self):
        
    #     logging.info(f"combining in tof mode:")

    #     logging.info(f"\tlooking at {NBR_TOF_RANGES} potential TOF ranges!")
    #     for _index, _widgets in enumerate(self.list_of_tof_ranges_ui):
    #         _use_it, _left_index, _right_index = _widgets.result
    #         logging.info(f"\ttof range index {_index}:")
    #         logging.info(f"\t\t{_use_it = }")
    #         logging.info(f"\t\t{_left_index = }")
    #         logging.info(f"\t\t{_right_index = }")




        # left_tof_index, right_tof_index = self.plot_tof_profile.result
        # logging.info(f"\tfrom index {left_tof_index} ({self.lambda_array_angstroms[left_tof_index]:0.3f} {ANGSTROMS})")
        # logging.info(f"\tto index {right_tof_index} ({self.lambda_array_angstroms[right_tof_index]:0.3f} {ANGSTROMS})")

        # master_3d_data_array = self.parent.master_3d_data_array

        # for _data_type in master_3d_data_array.keys():
        #     logging.info(f"\tbefore: {np.shape(master_3d_data_array[_data_type]) = }")
        #     new_master_3d_data_array = []
        #     for _data in master_3d_data_array[_data_type]:
        #         new_master_3d_data_array.append(np.mean(_data[left_tof_index:right_tof_index+1, :, :], axis=0))
        #     master_3d_data_array[_data_type] = np.array(new_master_3d_data_array)
        #     logging.info(f"\tafter: {np.shape(master_3d_data_array[_data_type]) = }")
        # self.parent.master_3d_data_array = master_3d_data_array
