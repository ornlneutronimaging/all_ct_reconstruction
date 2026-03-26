# generate code that list all the folders a user is allowed to access in '/SNS/VENUS/'
import os
import glob


def list_accessible_folders(base_path='/SNS/VENUS/'):
    
    list_ipts = glob.glob(os.path.join(base_path, 'IPTS-*'))
    accessible_folders = []
    for ipts_dir in list_ipts:
        # check that user can access the IPTS directory
        if os.access(ipts_dir, os.R_OK):
            accessible_folders.append(ipts_dir)
    return accessible_folders


if __name__ == "__main__":    
    folders = list_accessible_folders()
    for folder in folders:        
        print(folder)
        