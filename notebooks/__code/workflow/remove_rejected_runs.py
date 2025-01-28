import logging
import os
from tqdm import tqdm
import numpy as np

from __code.parent import Parent
from __code import DataType, Run
from __code.utilities.load import load_data_using_multithreading
from __code.utilities.files import retrieve_list_of_tif


class RemoveRejectedRuns(Parent):

    def run(self):
        logging.info(f"removing rejected runs:")
        list_of_sample_runs_to_reject = self.parent.list_of_sample_runs_to_reject_ui.value
        list_of_ob_runs_to_reject = self.parent.list_of_ob_runs_to_reject_ui.value
        logging.info(f"\tlist of sample runs to reject: {list_of_sample_runs_to_reject}")
        logging.info(f"\tlist of ob runs to reject: {list_of_ob_runs_to_reject}")

        for _run in self.parent.list_of_runs[DataType.sample].keys():
            if self.parent.list_of_runs[DataType.sample][_run][Run.use_it]:
                if _run in list_of_sample_runs_to_reject:
                    logging.info(f"\t\t rejecting sample run {_run}!")
                    self.parent.list_of_runs[DataType.sample][_run][Run.use_it] = False
                else:
                    self.parent.list_of_runs[DataType.sample][_run][Run.use_it] = True
                    self.parent.list_of_runs_to_use[DataType.sample].append(_run)

        for _run in self.parent.list_of_runs[DataType.ob].keys():
            if self.parent.list_of_runs[DataType.ob][_run][Run.use_it]:
                if _run in list_of_ob_runs_to_reject:
                    self.parent.list_of_runs[DataType.ob][_run][Run.use_it] = False
                    logging.info(f"\t\t rejecting ob run {_run}!")
                else:
                    self.parent.list_of_runs[DataType.ob][_run][Run.use_it] = True
                    self.parent.list_of_runs_to_use[DataType.ob].append(_run)
