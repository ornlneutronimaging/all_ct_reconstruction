import numpy as np


class MbirjaxReconstructionEvaluation:
    
    def __init__(self, data, list_angles_deg, reconstruction_parameters):
        self.corrected_array_log = data
        self.list_angles_rad = np.deg2rad(list_angles_deg)
        
        top_slice = reconstruction_parameters.get("top_slice", 0)
        bottom_slice = reconstruction_parameters.get("bottom_slice", self.corrected_array_log.shape[0])
            