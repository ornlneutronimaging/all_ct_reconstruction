import ipywidgets as widgets
from IPython.display import display
from IPython.display import HTML
from ipywidgets import interactive
import logging
import numpy as np

from __code.parent import Parent
from __code import OperatingMode, DataType
from __code import DEFAULT_OPERATING_MODE

from __code.workflow.checking_data import CheckingData
from __code.workflow.remove_rejected_runs import RemoveRejectedRuns
from __code.workflow.sort_runs import SortRuns
from __code.workflow.load import Load
from __code.workflow.tof_range_mode import TofRangeMode


class ModeSelection(Parent):

    def select(self):   
        self.mode_selection_ui = widgets.ToggleButtons(options=[OperatingMode.white_beam, 
                                                                OperatingMode.tof],
                                           value=DEFAULT_OPERATING_MODE)
        display(self.mode_selection_ui)
    
    def load(self):
        
        if self.mode_selection_ui.value == OperatingMode.white_beam:
            self.parent.operating_mode = OperatingMode.white_beam
        else:
            self.parent.operating_mode = OperatingMode.tof
        
        self.parent.configuration.operating_mode = self.mode_selection_ui.value

        logging.info(f"Working in {self.mode_selection_ui.value} mode")
        o_check = CheckingData(parent=self.parent)
        o_check.checking_minimum_requirements()
        if self.parent.minimum_requirements_met:

            o_rejected = RemoveRejectedRuns(parent=self.parent)
            o_rejected.run()

            o_sort = SortRuns(parent=self.parent)
            o_sort.run()

            combine_mode = (self.mode_selection_ui.value == OperatingMode.white_beam)
            o_load = Load(parent=self.parent)
            o_load.load_data(combine=combine_mode)

            if self.mode_selection_ui.value == OperatingMode.tof:

                master_3d_data_array = self.parent.master_3d_data_array[DataType.sample]
                logging.info(f"combining all the slices:")
                logging.info(f"\t{np.shape(master_3d_data_array) =}")
                merged_all_slices = np.sum(master_3d_data_array, axis=0)
                self.parent.data_3d_of_all_projections_merged = merged_all_slices
                logging.info(f"\t{np.shape(merged_all_slices) = }")
        
                o_load.load_spectra_file()

                # will display the profile of the region with lambda as x-axis
                self.parent.o_tof_range_mode = TofRangeMode(parent=self.parent)
                self.parent.o_tof_range_mode.run()

            # else:  # white beam mode
            #     self.parent.master_tof_3d_data_array = {'0': self.parent.master_3d_data_array}

        else:
            o_check.minimum_requirement_not_met()
