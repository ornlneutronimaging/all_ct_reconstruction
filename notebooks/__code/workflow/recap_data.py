import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
from IPython.core.display import HTML, display

from __code.parent import Parent
from __code import DataType, Run


class RecapData(Parent):

    final_list_of_runs = {DataType.sample: None,
                          DataType.ob: None}

    @staticmethod
    def is_pc_within_range(pc_value=0, pc_requested=0, threshold=1):
        if np.abs(pc_value - pc_requested) < threshold:
            return True
        else:
            return False

    def run(self):
        self.prepare_list_of_runs()
        self.display_list_of_runs()

    def prepare_list_of_runs(self):
        logging.info(f"Preparing list of runs to use:")
        pc_sample_requested, pc_ob_requested, pc_threshold = self.parent.selection_of_pc.result
        logging.info(f"\t{pc_sample_requested = }")
        logging.info(f"\t{pc_ob_requested = }")
        logging.info(f"\t{pc_threshold = }")

        list_of_runs = self.parent.list_of_runs
        
        final_list_of_sample_runs = []
        for _run in list_of_runs[DataType.sample].keys():
            logging.info(f"Working with {DataType.sample}")

            if list_of_runs[DataType.sample][_run][Run.use_it]:
    
                if list_of_runs[DataType.sample][_run][Run.use_it]:
                    _pc = list_of_runs[DataType.sample][_run][Run.proton_charge_c]
                    _angle = list_of_runs[DataType.sample][_run][Run.angle]
                    if RecapData.is_pc_within_range(pc_value=_pc,
                                                    pc_requested=pc_sample_requested,
                                                    threshold=pc_threshold):
                        final_list_of_sample_runs.append(_run)
                        logging.info(f"\t{_run} with pc of {_pc} ({_angle} degrees) is within the range !")
                        list_of_runs[DataType.sample][_run][Run.use_it] = True
                    else:
                        logging.info(f"\t{_run} with pc of {_pc} ({_angle} degrees) is not within the range !")
                        list_of_runs[DataType.sample][_run][Run.use_it] = False
                
                else:
                    logging.info(f"\t{_run} can not be used!")
                
        self.final_list_of_runs[DataType.sample] = final_list_of_sample_runs

        final_list_of_ob_runs = []
        for _run in list_of_runs[DataType.ob].keys():
            logging.info(f"Working with {DataType.ob}")

            if list_of_runs[DataType.ob][_run][Run.use_it]:
                _pc = list_of_runs[DataType.ob][_run][Run.proton_charge_c]
                if RecapData.is_pc_within_range(pc_value=_pc,
                                                pc_requested=pc_ob_requested,
                                                threshold=pc_threshold):
                    final_list_of_ob_runs.append(_run)
                    list_of_runs[DataType.ob][_run][Run.use_it] = True
                    logging.info(f"\t{_run} with pc of {_pc} is within the range !")

                else:
                    list_of_runs[DataType.ob][_run][Run.use_it] = False
                    logging.info(f"\t{_run} with pc of {_pc} is not within the range !")

            else:
                logging.info(f"\t{_run} can not be used!")

        self.final_list_of_runs[DataType.ob] = final_list_of_ob_runs
        self.parent.final_list_of_runs = self.final_list_of_runs
        self.parent.list_of_runs = list_of_runs

    def display_list_of_runs(self):

        sample_runs = widgets.VBox([
            widgets.Label("Sample"),
            widgets.SelectMultiple(options=self.final_list_of_runs[DataType.sample],
                                    layout=widgets.Layout(height="100%",
                                                            width='100%',
                                                            )),                                                       
        ],
        layout=widgets.Layout(width='200px',
                                height='300px'))
        self.parent.list_of_sample_runs_to_reject_ui = sample_runs.children[1]

        ob_runs = widgets.VBox([
            widgets.Label("OB"),
            widgets.SelectMultiple(options=self.final_list_of_runs[DataType.ob],
                                    layout=widgets.Layout(height="100%",
                                                            width='100%'))
        ],
        layout=widgets.Layout(width='200px',
                                height='300px'))
        self.parent.list_of_ob_runs_to_reject_ui = ob_runs.children[1]

        title = widgets.HTML("<b>Select any run(s) you want to exclude!:")

        hori_layout = widgets.HBox([sample_runs, ob_runs])
        verti_layout = widgets.VBox([title, hori_layout])
        display(verti_layout)
