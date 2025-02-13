import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import logging
from ipywidgets import interactive
from IPython.display import display
from IPython.core.display import HTML
import ipywidgets as widgets
from scipy.ndimage import median_filter
# from imars3d.backend.corrections.gamma_filter import gamma_filter
from tomopy.misc.corr import remove_outlier
from tomopy.misc.corr import remove_outlier_cuda
 
from __code import DataType, CleaningAlgorithm
from __code.config import clean_paras, NUM_THREADS, TOMOPY_REMOVE_OUTLIER_THRESHOLD_RATIO, TOMOPY_DIFF
from __code.parent import Parent
from __code.workflow.load import Load
from __code.workflow.export import Export
from __code.utilities.files import make_or_reset_folder
from __code.utilities.images import replace_pixels
from __code.utilities.logging import logging_3d_array_infos
from __code.workflow.data_handler import remove_negative_values


class ImagesCleaner(Parent): 

    """
    PixelCleaner Class: find the abnormal pixels (extremely low/high) and replaced by median value of the neighor matrix
    ===========
    to initiate cleaner: 
    data_cleaner = PixelCleaner(clean_paras, clean_path)
    [clean_paras]: parameters used for cleaning (dictionary)
    [clean_path]: the directory where save the cleaned data (if save) and logs (strings)
    ===========
    to clean a 2D image:
    cleaned_data = data_cleaner(orginal_im, save_file_name)
    [original_im]: the image need to be cleaned (M*M array)
    [save_file_name]: the cleaned tiff and its log will be saved in this name (strings)

    """
    low_gate = clean_paras['low_gate']
    high_gate = clean_paras['high_gate']
    r = clean_paras['correct_radius']
    CLEAN = clean_paras['if_clean']
    SAVE_CLEAN = clean_paras['if_save_clean']
    edge_nbr_pixels = clean_paras['edge_nbr_pixels']
    nbr_bins = clean_paras['nbr_bins']
            
    def settings(self):
        self.in_house_ui = widgets.Checkbox(value=False,
                                         description="In-house (histogram)")
        self.tomopy_ui = widgets.Checkbox(value=True,
                                        description="Tomopy (remove_outlier)")
        self.scipy_ui = widgets.Checkbox(value=False,
                                                 description="Scipy (median_filter)")
        v_box = widgets.VBox([self.in_house_ui, self.tomopy_ui, self.scipy_ui])
        display(v_box)

    def cleaning_setup(self):

        # update configuration
        list_algo = []
        if self.in_house_ui.value:
            list_algo.append(CleaningAlgorithm.in_house)
        if self.tomopy_ui.value:
            list_algo.append(CleaningAlgorithm.tomopy)
        if self.scipy_ui.value:
            list_algo.append(CleaningAlgorithm.scipy)
        
        self.parent.configuration.list_clean_algorithm = list_algo

        if self.in_house_ui.value:
            sample_data = np.array(self.parent.master_3d_data_array[DataType.sample])
            ob_data = np.array(self.parent.master_3d_data_array[DataType.ob])
            # if self.parent.master_3d_data_array[DataType.dc] is not None:
            #     dc_data = self.parent.master_3d_data_array[DataType.dc]
            # else:
            #     dc_data = None
            dc_data = None

            sample_histogram = sample_data.sum(axis=0)[self.edge_nbr_pixels: -self.edge_nbr_pixels,
                                                    self.edge_nbr_pixels: -self.edge_nbr_pixels]
            ob_histogram = ob_data.sum(axis=0)[self.edge_nbr_pixels: -self.edge_nbr_pixels,
                                            self.edge_nbr_pixels: -self.edge_nbr_pixels]

            # if dc_data is not None:
            #     dc_histogram = dc_data.sum(axis=0)[self.edge_nbr_pixels: -self.edge_nbr_pixels,
            #                     self.edge_nbr_pixels: -self.edge_nbr_pixels]

            nrows = 2 if dc_data is None else 3
            
            display(HTML("<h2> Histogram settings </h2>"))
            def plot_histogram(nbr_bins=100, nbr_exclude=1):
            
                fig, axs = plt.subplots(nrows=nrows, ncols=1)
                _, sample_bin_edges = np.histogram(sample_histogram.flatten(), bins=nbr_bins, density=False)
                axs[0].hist(sample_histogram.flatten(), bins=nbr_bins)
                axs[0].set_title('sample histogram')
                axs[0].set_yscale('log')
                axs[0].axvspan(sample_bin_edges[0], sample_bin_edges[nbr_exclude], facecolor='red', alpha=0.2)
                axs[0].axvspan(sample_bin_edges[-nbr_exclude-1], sample_bin_edges[-1], facecolor='red', alpha=0.2)
                
                _, ob_bin_edges = np.histogram(ob_histogram.flatten(), bins=nbr_bins, density=False)
                axs[1].hist(ob_histogram.flatten(), bins=nbr_bins)
                axs[1].set_title('ob histogram')
                axs[1].set_yscale('log')
                axs[1].axvspan(ob_bin_edges[0], ob_bin_edges[nbr_exclude], facecolor='red', alpha=0.2)
                axs[1].axvspan(ob_bin_edges[-nbr_exclude-1], ob_bin_edges[-1], facecolor='red', alpha=0.2)
                plt.tight_layout()
                plt.show()

                # if dc_data is not None:
                #     _, dc_bin_edges = np.histogram(dc_histogram.flatten(), bins=nbr_bins, density=False)
                #     axs[2].hist(dc_histogram.flatten(), bins=nbr_bins)
                #     axs[2].set_title('dc histogram')
                #     axs[2].set_yscale('log')
                #     axs[2].axvspan(dc_bin_edges[0], dc_bin_edges[nbr_exclude], facecolor='red', alpha=0.2)
                #     axs[2].axvspan(dc_bin_edges[-nbr_exclude-1], dc_bin_edges[-1], facecolor='red', alpha=0.2)
                #     plt.tight_layout()
                #     plt.show()

                return nbr_bins, nbr_exclude

            self.parent.display_histogram = interactive(plot_histogram,
                                                        nbr_bins = widgets.IntSlider(min=10,
                                                                                    max=1000,
                                                                                    value=100,
                                                                                    description='Nbr bins',
                                                                                    continuous_update=False),
                                                        nbr_exclude = widgets.IntSlider(min=0,
                                                                                        max=10,
                                                                                        value=1,
                                                                                        description='Bins to excl.',
                                                                                        continuous_update=False,
                                                                                        ),
                                                        )
            display(self.parent.display_histogram)

        if self.tomopy_ui.value: 

            display(HTML("<hr>"))
            display(widgets.HTML("<h2> Tomopy settings </h2>"))
            self.tomopy_diff = widgets.FloatSlider(min=1,
                                       max=100,
                                       value=20,
                                       description='Diff value')
            display(self.tomopy_diff)
        
    def cleaning(self, ignore_dc=False):

        # sample_data = self.parent.master_3d_data_array[DataType.sample]
        # ob_data = self.parent.master_3d_data_array[DataType.ob]
        # self.parent.master_3d_data_array = {DataType.sample: sample_data,
        #                                             DataType.ob: ob_data}

        if len(self.parent.configuration.list_clean_algorithm) > 0:
            sample_data = np.array(self.parent.master_3d_data_array[DataType.sample])
            self.parent.histogram_sample_before_cleaning = sample_data.sum(axis=0)[self.edge_nbr_pixels: -self.edge_nbr_pixels,
                                                self.edge_nbr_pixels: -self.edge_nbr_pixels]

        self.cleaning_by_histogram(ignore_dc=ignore_dc)
        self.cleaning_with_tomopy(ignore_dc=ignore_dc)
        self.cleaning_with_scipy(ignore_dc=ignore_dc)

    def cleaning_with_scipy(self, ignore_dc=False):
        """scipy"""

        if not self.scipy_ui.value:
            logging.info(f"cleaning using median filter: OFF")
            return  
        
        logging.info(f"cleaning using median filter ...")
        _size = (1, 3, 3)

        # sample
        logging_3d_array_infos(message="before scipy cleaning of sample", array=self.parent.master_3d_data_array[DataType.sample])
        self.parent.master_3d_data_array[DataType.sample] = np.array(median_filter(self.parent.master_3d_data_array[DataType.sample], size=_size))
        logging_3d_array_infos(message="after scipy cleaning of sample", array=self.parent.master_3d_data_array[DataType.sample])
        
        # ob
        logging_3d_array_infos(message="before scipy cleaning of ob", array=self.parent.master_3d_data_array[DataType.ob])       
        self.parent.master_3d_data_array[DataType.ob] = np.array(median_filter(self.parent.master_3d_data_array[DataType.ob], size=_size))
        logging_3d_array_infos(message="after scipy cleaning of ob", array=self.parent.master_3d_data_array[DataType.ob])       

        if not ignore_dc:
            if self.parent.list_of_images[DataType.dc]:
                logging_3d_array_infos(message="before scipy cleaning of dc", array=self.parent.master_3d_data_array[DataType.dc])       
                self.parent.master_3d_data_array[DataType.dc] = np.array(median_filter(self.parent.master_3d_data_array[DataType.dc], size=_size))
                logging_3d_array_infos(message="after scipy cleaning of dc", array=self.parent.master_3d_data_array[DataType.dc])       
        
        logging.info(f"cleaning using median filter ... done!")

    def cleaning_with_tomopy(self, ignore_dc=False):
        
        if not self.tomopy_ui.value:
            logging.info(f"cleaning using tomopy: OFF")
            return
    
        logging.info(f"cleaning using tomopy ...")
        sample_data = np.array(self.parent.master_3d_data_array[DataType.sample])

        logging_3d_array_infos(message="before tomopy cleaning of sample", array=self.parent.master_3d_data_array[DataType.sample])
        sample_data = np.array(self.parent.master_3d_data_array[DataType.sample])
        cleaned_sample = remove_outlier(sample_data, self.tomopy_diff.value, ncore=NUM_THREADS).astype(np.ushort)
        #cleaned_sample = gamma_filter(arrays=sample_data, diff_tomopy=self.tomopy_diff.value)
        self.parent.master_3d_data_array[DataType.sample] = cleaned_sample[:]
        logging_3d_array_infos(message="after tomopy cleaning of sample", array=self.parent.master_3d_data_array[DataType.sample])

        logging_3d_array_infos(message="before tomopy cleaning of ob", array=self.parent.master_3d_data_array[DataType.ob])
        ob_data = np.array(self.parent.master_3d_data_array[DataType.ob])
        cleaned_ob = remove_outlier(ob_data, self.tomopy_diff.value, ncore=NUM_THREADS).astype(np.ushort)
        # cleaned_ob = gamma_filter(arrays=ob_data, diff_tomopy=self.tomopy_diff.value)
        self.parent.master_3d_data_array[DataType.ob] = cleaned_ob[:]
        logging_3d_array_infos(message="after tomopy cleaning of ob", array=self.parent.master_3d_data_array[DataType.ob])

        if not ignore_dc:
            if self.parent.list_of_images[DataType.dc]:
                logging_3d_array_infos(message="before tomopy cleaning of dc", array=self.parent.master_3d_data_array[DataType.dc])
                dc_data = np.array(self.parent.master_3d_data_array[DataType.dc])
                cleaned_dc = remove_outlier(dc_data, self.tomopy_diff.value, ncore=NUM_THREADS).astype(np.ushort)
                # cleaned_dc = gamma_filter(arrays=dc_data, diff_tomopy=self.tomopy_diff.value)
                self.parent.master_3d_data_array[DataType.dc] = cleaned_dc[:]
                logging_3d_array_infos(message="after tomopy cleaning of dc", array=self.parent.master_3d_data_array[DataType.dc])

        logging.info(f"cleaning using tomopy ... done!")
            
    def cleaning_by_histogram(self, ignore_dc=False):

        if not self.in_house_ui.value:
            logging.info(f"cleaning by histogram: OFF")
            return

        logging.info(f"cleaning by histogram ...")
        self.nbr_bins, nbr_bins_to_exclude = self.parent.display_histogram.result

        # update configuration
        self.parent.configuration.histogram_cleaning_settings.nbr_bins = self.nbr_bins
        self.parent.configuration.histogram_cleaning_settings.bins_to_exclude = nbr_bins_to_exclude

        sample_data = self.parent.master_3d_data_array[DataType.sample]
        ob_data = self.parent.master_3d_data_array[DataType.ob]
        # dc_data = self.parent.master_3d_data_array[DataType.dc]

        if nbr_bins_to_exclude == 0:
            logging.info(f"0 bin selected, the raw data will be used!")

        else:         
            logging.info(f"user selected {nbr_bins_to_exclude} bins to exclude")
          
            logging.info(f"\t {np.shape(sample_data) = }")
            logging.info(f"\t {np.shape(ob_data) = }")

            logging.info(f"\tcleaning sample ...")
            cleaned_sample_data = []
            logging_3d_array_infos(message="before histogram cleaning of sample", array=self.parent.master_3d_data_array[DataType.sample])
            for _data in tqdm(sample_data):
                cleaned_im = replace_pixels(im=_data.copy(),
                                            nbr_bins=self.nbr_bins,
                                            low_gate=nbr_bins_to_exclude,
                                            high_gate=self.nbr_bins - nbr_bins_to_exclude,
                                            correct_radius=self.r)
                cleaned_sample_data.append(cleaned_im)          
            self.parent.master_3d_data_array[DataType.sample] = np.array(cleaned_sample_data)
            logging_3d_array_infos(message="after histogram cleaning of sample", array=self.parent.master_3d_data_array[DataType.sample])
            logging.info(f"\tcleaned sample!")

            logging.info(f"\tcleaning ob ...")
            logging_3d_array_infos(message="before histogram cleaning of ob", array=self.parent.master_3d_data_array[DataType.ob])
            cleaned_ob_data = []
            for _data in tqdm(ob_data):
                cleaned_im = replace_pixels(im=_data.copy(),
                                            nbr_bins=self.nbr_bins,
                                            low_gate=nbr_bins_to_exclude,
                                            high_gate=self.nbr_bins - nbr_bins_to_exclude,
                                            correct_radius=self.r)
                cleaned_ob_data.append(cleaned_im)          
            self.parent.master_3d_data_array[DataType.ob] = np.array(cleaned_ob_data)
            logging_3d_array_infos(message="after histogram cleaning of ob", array=self.parent.master_3d_data_array[DataType.ob])
            logging.info(f"\tcleaned ob!")

        # dislay result of cleaning
        sample_data = np.array(self.parent.master_3d_data_array[DataType.sample])
        ob_data = np.array(self.parent.master_3d_data_array[DataType.ob])

        sample_histogram = sample_data.sum(axis=0)[self.edge_nbr_pixels: -self.edge_nbr_pixels,
                                                self.edge_nbr_pixels: -self.edge_nbr_pixels]
        ob_histogram = ob_data.sum(axis=0)[self.edge_nbr_pixels: -self.edge_nbr_pixels,
                                        self.edge_nbr_pixels: -self.edge_nbr_pixels]

        
        nbr_bins = self.nbr_bins

        fig, axs = plt.subplots(nrows=2, ncols=1)

        flatten_sample_histogram = sample_histogram.flatten()
        _, sample_bin_edges = np.histogram(flatten_sample_histogram, bins=nbr_bins, density=False)
        axs[0].hist(flatten_sample_histogram, bins=nbr_bins)
        axs[0].set_title('cleaned sample histogram')
        axs[0].set_yscale('log')
        
        flatten_ob_histogram = ob_histogram.flatten()
        _, ob_bin_edges = np.histogram(flatten_ob_histogram, bins=nbr_bins, density=False)
        axs[1].hist(flatten_ob_histogram, bins=nbr_bins)
        axs[1].set_title('cleaned ob histogram')
        axs[1].set_yscale('log')
        plt.tight_layout()
        plt.show()

        logging.info(f"cleaning by histogram ... done!")

    def select_export_folder(self):
        o_load = Load(parent=self.parent)
        o_load.select_folder(data_type=DataType.cleaned_images)
    
    def export_clean_images(self):
        logging.info(f"Exporting the cleaned images")
        logging.info(f"\tfolder selected: {self.parent.working_dir[DataType.cleaned_images]}")

        master_3d_data = self.parent.master_3d_data_array_cleaned

        master_base_folder_name = f"{os.path.basename(self.parent.working_dir[DataType.sample])}_cleaned"
        full_output_folder = os.path.join(self.parent.working_dir[DataType.cleaned_images],
                                          master_base_folder_name)

        # sample
        logging.info(f"working with sample:")
        sample_full_output_folder = os.path.join(full_output_folder, "sample")
        logging.info(f"\t {sample_full_output_folder =}")
        make_or_reset_folder(sample_full_output_folder)

        o_export = Export(image_3d=master_3d_data[DataType.sample],
                          output_folder=sample_full_output_folder)
        o_export.run()

        # ob
        logging.info(f"working with ob:")
        ob_full_output_folder = os.path.join(full_output_folder, 'ob')
        logging.info(f"\t {ob_full_output_folder =}")
        make_or_reset_folder(ob_full_output_folder)

        o_export = Export(image_3d=master_3d_data[DataType.ob],
                          output_folder=ob_full_output_folder)
        o_export.run()
        
    def remove_outliers(self, data_3d):
        _data_3d = remove_outlier_cuda(data_3d, TOMOPY_DIFF)
        _data_3d = remove_negative_values(_data_3d)
        return _data_3d[:]
    