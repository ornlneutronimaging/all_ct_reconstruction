import h5py


def get_proton_charge(nexus, units='pc'):
    if nexus is None:
        return None
    
    try:
        with h5py.File(nexus, 'r') as hdf5_data:
            proton_charge = hdf5_data["entry"]["proton_charge"][0]
            if units == 'c':
                return proton_charge/1e12
            return proton_charge
    except FileNotFoundError:
        return None
    

def get_frame_number(nexus):
    if nexus is None:
        return None

    try:
        with h5py.File(nexus, 'r') as hdf5_data:
            frame_number = hdf5_data["entry"]['DASlogs']['BL10:Det:PIXELMAN:ACQ:NUM']['value'][:][-1]
            return frame_number
    except KeyError:
        return None
    