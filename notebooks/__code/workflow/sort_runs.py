import logging

from __code.parent import Parent
from __code import DataType, Run


class SortRuns(Parent):

    def run(self):

        logging.info(f"Sorting the runs by increasing angle value!")
        list_of_angles = []
        list_of_runs = self.parent.list_of_runs_to_use[DataType.sample]
        for _run in list_of_runs:
            list_of_angles.append(self.parent.list_of_runs[DataType.sample][_run][Run.angle])

        # sort the angles and sort the runs the same way
        index_sorted = sorted(range(len(list_of_angles)), key=lambda k: list_of_angles[k])
        list_of_runs_sorted = [list_of_runs[_index] for _index in index_sorted]
        list_of_angles_sorted = [list_of_angles[_index] for _index in index_sorted]

        logging.info(f"\t{list_of_runs_sorted = }")
        logging.info(f"\t{list_of_angles_sorted = }")

        self.parent.list_of_runs_to_use[DataType.sample] = list_of_runs_sorted
        self.parent.list_of_angles_to_use_sorted = list_of_angles_sorted
        