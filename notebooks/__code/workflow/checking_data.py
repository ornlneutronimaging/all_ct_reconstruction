import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
from IPython.core.display import HTML, display

from __code.parent import Parent
from __code.config import PROTON_CHARGE_TOLERANCE_C
from __code import DataType, Run
from __code.utilities.files import retrieve_list_of_runs, retrieve_list_of_tif, get_angle_value
from __code.utilities.nexus import get_proton_charge, get_frame_number
from __code.utilities.math import calculate_most_dominant_int_value_from_list


class CheckingData(Parent):

    list_of_runs = {DataType.sample: None,
                    DataType.ob: None}
    list_of_metadata = {}

    # def get_angle_value(self, run_full_path=None):
    #     """ extract the rotation angle value from a string name looking like 
    #     Run_####_20240927_date_..._148_443_######_<file_index>.tif
    #     """
    #     list_tiff = retrieve_list_of_tif(run_full_path)
    #     first_tiff = list_tiff[0]
    #     list_part = first_tiff.split("_")
    #     return f"{list_part[-4]}.{list_part[-3]}"

    def run(self):

        # retrieve the full list of runs found in the top folder
        self.retrieve_runs()

        # check empty runs
        self.reject_empty_runs()

        # retrieve proton charge of runs
        self.retrieve_proton_charge()

        # retrieve rotation angle
        self.retrieve_rotation_angle()

        # retrieve frame number
        self.retrieve_frame_number()

        # display graph
        self.display_graph()

    def retrieve_frame_number(self):
        logging.info(f"Retrieving frame numbers:")
        self.parent.at_least_one_frame_number_not_found = False
        for _data_type in self.parent.list_of_runs.keys():
            logging.info(f"\t{_data_type}:")
            list_of_frame_number = []
            for _run in self.parent.list_of_runs[_data_type]:
                # _, number = os.path.basename(_run).split("_")
                # nexus_path = os.path.join(top_nexus_path, f"{self.parent.instrument}_{number}.nxs.h5")
                nexus_path = self.parent.list_of_runs[_data_type][_run][Run.nexus]
                frame_number = get_frame_number(nexus_path)
                list_of_frame_number.append(frame_number)
                self.parent.list_of_runs[_data_type][_run][Run.frame_number] = frame_number
                if not frame_number:
                    self.parent.at_least_one_frame_number_not_found = True
                
            logging.info(f"\t\t{list_of_frame_number}")

    def retrieve_rotation_angle(self):
        list_of_sample_runs = self.parent.list_of_runs[DataType.sample]
        for _run in list_of_sample_runs.keys():
            if list_of_sample_runs[_run][Run.use_it]:
                angle_value = get_angle_value(run_full_path=list_of_sample_runs[_run][Run.full_path])
                self.parent.list_of_runs[DataType.sample][_run][Run.angle] = angle_value

    def retrieve_runs(self):
        ''' retrieve the full list of runs in the top folder of sample and ob '''
        
        logging.info(f"Retrieving runs:")
        for _data_type in self.list_of_runs.keys():
            list_of_runs = retrieve_list_of_runs(top_folder=self.parent.working_dir[_data_type])
            if len(list_of_runs) == 0:
                display(HTML(f"<font color=red>Found 0 {_data_type} runs in {self.parent.working_dir[_data_type]}</font>"))
                raise ValueError("Missing files !")
            
            logging.info(f"\tfound {len(list_of_runs)} {_data_type} runs")
            
            top_nexus_path = self.parent.working_dir[DataType.nexus]
            for _run in list_of_runs:
                _, number = os.path.basename(_run).split("_")
                nexus_path = os.path.join(top_nexus_path, f"{self.parent.instrument}_{number}.nxs.h5")
                self.parent.list_of_runs[_data_type][os.path.basename(_run)] = {Run.full_path: _run,
                                                                                Run.proton_charge_c: None,
                                                                                Run.use_it: True,
                                                                                Run.angle: None,
                                                                                Run.nexus: nexus_path,
                                                                                Run.frame_number: None,
                                                                                }

    def reject_empty_runs(self):

        logging.info(f"Rejecting empty runs:")
        for _data_type in self.parent.list_of_runs.keys():
            list_of_runs = list(self.parent.list_of_runs[_data_type].keys())

            list_of_runs_to_keep = []
            list_of_runs_to_remove = []

            for _run in list_of_runs:

                _run_full_path = self.parent.list_of_runs[_data_type][_run][Run.full_path]
                list_tif = retrieve_list_of_tif(_run_full_path)
                if len(list_tif) > 0:
                    list_of_runs_to_keep.append(_run_full_path)
                    self.parent.list_of_runs[_data_type][_run][Run.use_it] = True
                else:
                    list_of_runs_to_remove.append(_run)
                    self.parent.list_of_runs[_data_type][_run][Run.use_it] = False

            logging.info(f"\trejected {len(list_of_runs_to_remove)} {_data_type} runs")
            logging.info(f"\t -> {[os.path.basename(_file) for _file in list_of_runs_to_remove]}")

    def retrieve_proton_charge(self):
        logging.info(f"Retrieving proton charge values:")
        top_nexus_path = self.parent.working_dir[DataType.nexus]
        list_proton_charge_c = {DataType.sample: [],
                                DataType.ob: []} 
        
        min_proton_charge_c = {DataType.sample: None,
                               DataType.ob: None}
        
        max_proton_charge_c = {DataType.sample: None,
                               DataType.ob: None}

        for _data_type in self.parent.list_of_runs.keys():
            _list_proton_charge = []
            for _run in self.parent.list_of_runs[_data_type]:
                nexus_path = self.parent.list_of_runs[_data_type][_run][Run.nexus]
                proton_charge = get_proton_charge(nexus_path)
                _list_proton_charge.append(proton_charge)
                self.list_of_metadata[_run] = proton_charge
                self.parent.list_of_runs[_data_type][_run][Run.proton_charge_c] = proton_charge/1e12
            list_proton_charge_c[_data_type] = [_pc/1e12 for _pc in _list_proton_charge]
            logging.info(f"\t{_data_type}: {list_proton_charge_c[_data_type]}")

            min_proton_charge_c[_data_type] = min(list_proton_charge_c[_data_type]) - 1
            max_proton_charge_c[_data_type] = max(list_proton_charge_c[_data_type]) + 1

        self.list_proton_charge_c = list_proton_charge_c
        self.min_proton_charge_c = min_proton_charge_c
        self.max_proton_charge_c = max_proton_charge_c
  
    def display_graph(self):
        
        default_sample_proton_charge = calculate_most_dominant_int_value_from_list(self.list_proton_charge_c[DataType.sample])
        default_ob_proton_charge = calculate_most_dominant_int_value_from_list(self.list_proton_charge_c[DataType.ob])

        def plot_proton_charges(sample_proton_charge_value, ob_proton_charge_value, proton_charge_threshold):        
            fig, axs = plt.subplots(nrows=1, ncols=1)
            axs.set_title("Proton charge (C) of selected runs")
            axs.plot(self.list_proton_charge_c[DataType.sample], 'g+', label=DataType.sample)
            axs.plot(self.list_proton_charge_c[DataType.ob], 'bo', label=DataType.ob)
            axs.set_xlabel("file index")
            axs.set_ylabel("proton charge (C)")
            axs.legend()
    
            axs.axhline(sample_proton_charge_value, linestyle='--', color='green')
            sample_proton_charge_range = [sample_proton_charge_value + proton_charge_threshold,
                                sample_proton_charge_value - proton_charge_threshold]
            axs.axhspan(sample_proton_charge_range[0], 
                        sample_proton_charge_range[1], facecolor='green', alpha=0.2)

            axs.axhline(ob_proton_charge_value, linestyle='--', color='blue')
            ob_proton_charge_range = [ob_proton_charge_value + proton_charge_threshold,
                                    ob_proton_charge_value - proton_charge_threshold]
            axs.axhspan(ob_proton_charge_range[0], 
                        ob_proton_charge_range[1], facecolor='blue', alpha=0.2)

            plt.show()

            return sample_proton_charge_value, ob_proton_charge_value, proton_charge_threshold

        self.parent.selection_of_pc = interactive(plot_proton_charges,
                            sample_proton_charge_value = widgets.FloatSlider(min=self.min_proton_charge_c[DataType.sample],
                                                                    max=self.max_proton_charge_c[DataType.sample],
                                                                    value=default_sample_proton_charge,
                                                                    description='sample pc',
                                                                    continuous_update=True),
                            ob_proton_charge_value = widgets.FloatSlider(min=self.min_proton_charge_c[DataType.ob],
                                                                    max=self.max_proton_charge_c[DataType.ob],
                                                                    value=default_ob_proton_charge,
                                                                    description='ob pc',
                                                                    continuous_update=True),
                            proton_charge_threshold = widgets.FloatSlider(min=0.0001,
                                                                        max=1,
                                                                        description='threshold',
                                                                        value=PROTON_CHARGE_TOLERANCE_C,
                                                                        continuous_update=True),
                                                                        )
        display(self.parent.selection_of_pc)

    def checking_minimum_requirements(self):
        """at least 1 OB and 3 samples selected"""
        logging.info(f"Checking minimum requirements:")
                
        list_ob = self.parent.list_of_ob_runs_to_reject_ui.options
        list_ob_selected = self.parent.list_of_ob_runs_to_reject_ui.value

        # at least 1 OB
        if len(list_ob) == 0:
            logging.info(f"\tno OB available. BAD!")
            self.parent.minimum_requirements_met = False
            return
        
        # at least 1 OB to keep
        if len(list_ob) == len(list_ob_selected):
            logging.info(f"\tnot keeping any OB. BAD!")
            self.parent.minimum_requirements_met = False
            return
        
        list_sample = self.parent.list_of_sample_runs_to_reject_ui.options
        list_sample_selected = self.parent.list_of_sample_runs_to_reject_ui.value
        
        # at least 3 projections
        if len(list_sample) < 3:
            logging.info(f"\tless than 3 sample runs available. BAD!")
            self.parent.minimum_requirements_met = False
            return
        
        if len(list_sample) == len(list_sample_selected):
            logging.info(f"\tnot keeping any sample run. BAD!")
            self.parent.minimum_requirements_met = False
            return
        
        logging.info(f"At least 1 OB and 3 sample runs. GOOD!")
        self.parent.minimum_requirements_met = True

    def minimum_requirement_not_met(self):
        display(HTML(f"<font color=red><b>STOP!</b> Make sure you have at least 3 sample and 1 OB selected!</font>"))
        logging.info(f"Minimum requirement not met!")
