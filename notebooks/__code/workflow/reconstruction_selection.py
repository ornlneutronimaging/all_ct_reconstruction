import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML
from ipywidgets import interactive
import logging
import numpy as np

from __code.parent import Parent
from  __code.utilities.general import retrieve_list_class_attributes_name
from __code import DEFAULT_RECONSTRUCTION_ALGORITHM
from __code.utilities.configuration_file import ReconstructionAlgorithm


class ReconstructionSelection(Parent):

    def select(self):   

        # get all the attribute names of the ReconstructionAlgorithm class
        list_algo = retrieve_list_class_attributes_name(ReconstructionAlgorithm)

        label = widgets.HTML("<font size=5 color=blue>Select reconstruction algorithm(s)</font>")
        self.multi_reconstruction_selection_ui = widgets.SelectMultiple(options=list_algo,
                                                                        rows=len(list_algo),
                                                                        description="",
                                                                        value=DEFAULT_RECONSTRUCTION_ALGORITHM,
        )

        self.multi_reconstruction_selection_ui.observe(self.on_change, names='value')

        display(widgets.VBox([label, self.multi_reconstruction_selection_ui]))
      
    def on_change(self, change):
        selected_values = change['new']
        self.parent.configuration.reconstruction_algorithm = selected_values
        logging.info(f"selected reconstruction algorithm: {selected_values}")
        