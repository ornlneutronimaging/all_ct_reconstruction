# Step 1: CCD Image Preparation for CT Reconstruction - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [System Requirements](#system-requirements)
4. [Workflow Architecture](#workflow-architecture)
5. [Detailed Step-by-Step Guide](#detailed-step-by-step-guide)
6. [API Reference](#api-reference)
7. [Data Structures](#data-structures)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Overview

The `step1_prepare_CCD_images.ipynb` notebook is the first stage in the CT reconstruction pipeline for CCD detector data. It provides an interactive Jupyter notebook interface for preparing raw CCD projection images for tomographic reconstruction.

### Purpose
- Load and organize raw CCD projection images
- Apply normalization using Open Beam (OB) and Dark Current (DC) images
- Perform image preprocessing (cropping, rotation, cleaning)
- Calculate center of rotation and tilt correction
- Export preprocessed data and configuration files for reconstruction

### Key Features
- Interactive widget-based UI for parameter selection
- Support for both normalized and unnormalized data
- Flexible angle retrieval methods (metadata, filename, or manual)
- Multiple preprocessing options (outlier removal, stripe removal, etc.)
- Test reconstruction capabilities before full processing
- Configurable reconstruction algorithm selection (SVMBIR, MBIRJAX, FBP)

---

## Prerequisites

### Required Knowledge
- Basic understanding of CT reconstruction principles
- Familiarity with Jupyter notebooks
- Understanding of neutron imaging data formats
- Knowledge of TIFF image formats

### Software Dependencies

See `requirements.txt` for full dependency list. Key packages include:

```
- Python 3.10+
- hsnt (https://github.com/cabouman/hsnt)
- iMars3D (for image processing)
- algotom (stripe removal algorithms)
- tomopy (reconstruction algorithms)
- astra-toolbox (reconstruction)
- neutompy (tilt correction)
- h5py (NeXus file reading)
- dxchange (image I/O)
- scikit-image
- opencv-python
```

### Installation Notes
```bash
# Clone iMars3D
git clone https://github.com/ornlneutronimaging/iMars3D
cd iMars3D
pip install -e .

# Install via micromamba
micromamba install algotom pydantic tqdm jupyter h5py param dxchange tomopy
micromamba install matplotlib pandas
micromamba install -c simpleitk simpleitk
micromamba install ipython numexpr astropy tifffile mkl_fft
micromamba install -c astra-toolbox astra-toolbox

# Install via pip
pip install neutronbraggedge
pip install opencv-python read-roi

# Install modified NeuTomPy
git clone https://github.com/dmici/NeuTomPy-toolbox
cd NeuTomPy-toolbox
pip install .
```

**Warning**: The workflow requires a locally modified version of NeuTomPy with interactivity removed from tilt calculation.

---

## System Requirements

### Directory Structure
The workflow expects the following directory structure:

```
/SNS/{INSTRUMENT}/{IPTS}/
├── raw/
│   ├── ct_scans/          # Sample projection images
│   ├── ob/                # Open beam images
│   └── dc/                # Dark current images
├── nexus/                 # NeXus metadata files
└── shared/
    └── processed_data/    # Output directory
```

### Data Format
- **Sample Images**: TIFF format, one image per rotation angle
- **Open Beam**: TIFF format, multiple images for averaging
- **Dark Current**: TIFF format, multiple images for averaging
- **Metadata**: NeXus HDF5 files (optional, for angle retrieval)

---

## Workflow Architecture

### Class: `Step1PrepareCcdImages`

The main class orchestrating the entire workflow, located in `__code/step1_prepare_ccd_images.py`.

#### Operating Mode
```python
MODE = OperatingMode.white_beam
```
This workflow operates in white beam mode (as opposed to Time-of-Flight mode).

#### Key Attributes

```python
# Data storage
master_3d_data_array = {
    DataType.sample: None,  # [angle, y, x]
    DataType.ob: None,
    DataType.dc: None
}

# Working directories
working_dir = {
    DataType.sample: "",
    DataType.ob: "",
    DataType.nexus: "",
    DataType.cleaned_images: "",
    DataType.normalized: "",
    DataType.processed: "",
}

# Processing results
normalized_images = None          # After normalization
normalized_images_log = None      # After log conversion
corrected_images = None           # After chips correction
sinogram_normalized_images_log = None  # Sinograms

# Metadata
final_list_of_angles = None       # Rotation angles for each projection
center_of_rotation = None         # Calculated center of rotation
crop_region = {'left': None, 'right': None, 'top': None, 'bottom': None}
```

### Workflow Components

The workflow is modular, using specialized classes for each operation:

| Component | Class | Purpose |
|-----------|-------|---------|
| Data Loading | `Load` | Select and load images from disk |
| Data Combination | `CombineObDc` | Average OB/DC images |
| Image Cleaning | `ImagesCleaner` | Remove outliers and hot/cold pixels |
| Normalization | `Normalization` | Apply OB/DC normalization |
| Cropping | `Crop` | Crop images to region of interest |
| Rotation | `Rotate` | Rotate images to correct orientation |
| Rebinning | `Rebin` | Reduce image size by binning |
| Strip Removal | `RemoveStrips` | Remove vertical/horizontal stripes |
| Tilt Correction | `CenterOfRotationAndTilt` | Calculate and apply tilt |
| Center of Rotation | `CenterOfRotationAndTilt` | Find rotation center |
| Test Reconstruction | `TestReconstruction` | Quick reconstruction test |
| SVMBIR Handler | `SvmbirHandler` | Configure SVMBIR reconstruction |
| FBP Handler | `FbpHandler` | Configure FBP reconstruction |
| Visualization | `Visualization` | Display images and results |
| Export | `ExportExtra` | Save configuration and data |

---

## Detailed Step-by-Step Guide

### Section 1: Initialization and Setup

#### Cell 1: Import and Setup
```python
import warnings
warnings.filterwarnings('ignore')

from __code.step1_prepare_ccd_images import Step1PrepareCcdImages
from __code import system

# Set working directory for specific IPTS and instrument
system.System.select_working_dir(ipts="IPTS-33767", instrument="CG1D")
from __code.__all import custom_style
custom_style.style()

Step1PrepareCcdImages.legend()
```

**What it does:**
- Imports required modules
- Sets the working directory based on IPTS number and instrument
- Displays a legend explaining color coding:
  - **Red**: Mandatory steps
  - **Orange**: Optional but recommended
  - **Purple**: Optional enhancement steps

**Parameters:**
- `ipts`: IPTS experiment number (e.g., "IPTS-33767")
- `instrument`: Instrument name (e.g., "CG1D", "VENUS")

---

### Section 2: Input Sample Folder (MANDATORY)

#### Cell 2: Select Sample Images
```python
o_white_beam = Step1PrepareCcdImages(system=system)
o_white_beam.select_top_sample_folder()
```

**What it does:**
- Creates main workflow object
- Opens file browser to select folder containing projection images
- Scans folder and loads list of image files
- Updates `list_of_images[DataType.sample]`

**Expected Input:**
- Folder containing TIFF images, one per rotation angle
- Typical naming: `image_000.tif`, `image_001.tif`, etc.

**Output:**
- Populates internal list of sample images
- Displays number of images found

---

### Section 3: Input Open Beam Images (OPTIONAL)

#### Cell 3: Select Open Beam Images
```python
o_white_beam.select_ob_images()
```

**What it does:**
- Opens file browser to select individual OB images
- Multiple images will be averaged to reduce noise
- Updates `list_of_images[DataType.ob]`

**When to skip:**
- Data is already normalized
- No OB images available

**Best Practice:**
- Select 10-20 OB images for good statistics
- OB images should be acquired under same conditions as sample

---

### Section 4: Input Dark Current Images (OPTIONAL)

#### Cell 4: Select Dark Current Images
```python
o_white_beam.select_dc_images()
```

**What it does:**
- Opens file browser to select individual DC images
- Multiple images will be averaged
- Updates `list_of_images[DataType.dc]`

**When to skip:**
- No OB images selected (DC is only used with OB)
- DC correction not needed

**Note:** DC selection is ignored if OB was not selected.

---

### Section 5: Projection Angle Retrieval (MANDATORY)

#### Cell 5: How to Retrieve Angles
```python
o_white_beam.how_to_retrieve_angle_value()
```

**What it does:**
- Presents three options for angle retrieval:
  1. **From NeXus metadata**: Read angles from HDF5 metadata files
  2. **From filename**: Parse angles from image filenames
  3. **Manual entry**: Manually specify angle range

**Method 1: From NeXus Metadata**
```python
o_white_beam.retrieve_angle_value()
```
- Requires NeXus files in the `nexus/` directory
- Automatically extracts rotation angles from metadata
- Most reliable method if metadata is available

**Method 2: From Filename**
```python
o_white_beam.retrieve_angle_value()
```
- Configure filename parsing pattern
- Example: `sample_045deg.tif` → extract "045"
- Useful when metadata unavailable

**Method 3: Manual Entry**
- Specify start angle, end angle, and number of steps
- Assumes evenly spaced angles
- Least accurate but always available

#### Cell 6: Test Angle Values
```python
o_white_beam.testing_angle_values()
```

**What it does:**
- Displays retrieved angles in a table
- Allows verification of angle values
- Shows angle vs. image index plot

---

### Section 6: Data Loading Options

#### Cell 7: Use All or Fraction
```python
o_white_beam.use_all_or_fraction()
```

**What it does:**
- Choose whether to use all images or a subset
- Useful for:
  - Quick testing with smaller dataset
  - Memory constraints
  - Specific angle range selection

#### Cell 8: Select Percentage (if fraction chosen)
```python
o_white_beam.select_percentage_of_data_to_use()
```

**What it does:**
- Slider to select percentage of data (1-100%)
- Displays number of images that will be loaded

---

### Section 7: Load Data (MANDATORY)

#### Cell 9: Load Images
```python
o_white_beam.load_data()
```

**What it does:**
- Loads all selected images into memory
- Creates `master_3d_data_array[DataType.sample]` with shape `[n_angles, height, width]`
- Loads OB and DC images if selected
- Associates angles with images
- Progress bar shows loading status

**Memory Note:** 
- Loading all images can consume significant memory
- For 500 images of 2048x2048 pixels (float32): ~8 GB RAM

**Creates:**
- `master_3d_data_array[DataType.sample]`: 3D array of sample projections
- `master_3d_data_array[DataType.ob]`: OB images
- `master_3d_data_array[DataType.dc]`: DC images
- `final_list_of_angles`: Array of rotation angles in degrees
- `final_list_of_angles_rad`: Array of rotation angles in radians

---

### Section 8: Visualization

#### Cell 10: How to Visualize
```python
o_white_beam.how_to_visualize()
```

**What it does:**
- Select visualization type:
  - **Image viewer**: Browse through projections
  - **Profile plot**: Intensity profiles
  - **Histogram**: Intensity distribution

#### Cell 11: Visualize Raw Data
```python
o_white_beam.visualize_raw_data()
```

**What it does:**
- Opens interactive viewer for raw projections
- Features:
  - Slider to navigate through angles
  - Zoom and pan
  - Colormap selection
  - Intensity histogram
  - Coordinate display

---

### Section 9: Image Exclusion (OPTIONAL)

#### Cell 12: Selection Mode
```python
o_white_beam.selection_mode()
```

**What it does:**
- Choose exclusion method:
  - **Manual selection**: Pick specific images to exclude
  - **Angle range**: Exclude images in angle range

#### Cell 13: Process Exclusion
```python
o_white_beam.process_exclusion_mode()
```

**What it does:**
- Interactive UI to select images for exclusion
- Displays thumbnails or list of images
- Check boxes to mark images for removal

#### Cell 14: Apply Exclusion
```python
o_white_beam.exclude_this_list_of_images()
```

**What it does:**
- Removes selected images from `master_3d_data_array`
- Updates angle list
- Updates image count

**Use Cases:**
- Remove corrupted images
- Exclude incomplete rotations
- Remove images with artifacts

---

### Section 10: Pre-processing Crop (OPTIONAL)

#### Cell 15: Crop Settings
```python
o_white_beam.pre_processing_crop_settings()
```

**What it does:**
- Interactive ROI selector on representative image
- Drag rectangle to define crop region
- Reduces data size early in pipeline

#### Cell 16: Apply Pre-processing Crop
```python
o_white_beam.pre_processing_crop()
```

**What it does:**
- Crops all images to selected ROI
- Updates `master_3d_data_array` shape
- Reduces memory usage and processing time

**When to use:**
- Large empty regions in images
- Want to focus on specific area
- Memory constraints

---

### Section 11: Image Cleaning - Outlier Removal (OPTIONAL but RECOMMENDED)

#### Cell 17: Cleaning Settings
```python
o_white_beam.clean_images_settings()
```

**What it does:**
- Configure outlier removal parameters:
  - **Method**: Median filter, mean filter, or threshold
  - **Kernel size**: Size of filter window
  - **Threshold**: Number of standard deviations

#### Cell 18: Cleaning Setup
```python
o_white_beam.clean_images_setup()
```

**What it does:**
- Preview cleaning effect on test image
- Adjust parameters interactively
- Shows before/after comparison

#### Cell 19: Apply Cleaning
```python
o_white_beam.clean_images()
```

**What it does:**
- Applies outlier removal to all images
- Replaces hot/cold pixels with local median
- Updates `master_3d_data_array`
- Progress bar shows status

**Removes:**
- Hot pixels (abnormally bright)
- Cold pixels (abnormally dark)
- Gamma rays hits
- Detector noise spikes

---

### Section 12: Visualization After Cleaning

#### Cell 20: Visualization Settings
```python
o_white_beam.how_to_visualize_after_cleaning()
```

#### Cell 21: Visualize Cleaned Data
```python
o_white_beam.visualize_cleaned_data()
```

**What it does:**
- Compare raw vs. cleaned images side-by-side
- Verify cleaning effectiveness
- Identify remaining artifacts

---

### Section 13: Normalization (OPTIONAL but RECOMMENDED)

#### Cell 22: Normalization Settings
```python
o_white_beam.normalization_settings()
```

**What it does:**
- Configure normalization options:
  - **Background ROI**: Select region for background subtraction
  - **Normalization method**: Standard or custom

**Only available if OB images were selected.**

#### Cell 23: Select ROI for Normalization
```python
o_white_beam.normalization_select_roi()
```

**What it does:**
- Interactive ROI selector for background region
- Select area outside sample for beam monitoring
- Optional - can skip for standard normalization

#### Cell 24: Perform Normalization
```python
o_white_beam.normalization()
```

**What it does:**
- Averages OB and DC images
- Performs normalization: `I_norm = (Sample - DC) / (OB - DC)`
- Creates `normalized_images` array
- Normalizes to range [0, 1]

**Formula:**
```
I_normalized = (I_sample - I_dc) / (I_ob - I_dc)
```

**Benefits:**
- Corrects for beam intensity variations
- Removes detector non-uniformity
- Essential for quantitative analysis

---

### Section 14: Visualization of Normalization

#### Cell 25: Visualization Settings
```python
o_white_beam.visualization_normalization_settings()
```

#### Cell 26: Visualize Normalization
```python
o_white_beam.visualize_normalization()
```

**What it does:**
- Side-by-side comparison of cleaned vs. normalized
- Verify normalization quality
- Check for artifacts introduced by normalization

---

### Section 15: Export Normalized Images (OPTIONAL)

#### Cell 27: Select Export Folder
```python
o_white_beam.select_export_normalized_folder()
```

#### Cell 28: Export Normalized
```python
o_white_beam.export_normalized_images()
```

**What it does:**
- Saves normalized images as TIFF stack
- Useful for external processing
- Can be reloaded without repeating normalization

---

### Section 16: Rebinning (OPTIONAL)

#### Cell 29: Rebin Settings
```python
o_white_beam.rebin_settings()
```

**What it does:**
- Configure binning factors (2x2, 4x4, etc.)
- Reduces image size by averaging pixels
- Select whether to bin before or after normalization

#### Cell 30: Apply Rebinning Before Normalization
```python
o_white_beam.rebin_before_normalization()
```

**Or After Normalization:**
```python
o_white_beam.rebin_after_normalization()
```

**What it does:**
- Bins images by averaging adjacent pixels
- Reduces noise at cost of spatial resolution
- Speeds up subsequent processing

**Example:** 2x2 binning on 2048x2048 → 1024x1024

#### Cell 31: Visualize Rebinned Data
```python
o_white_beam.visualize_rebinned_data(before_normalization=False)
```

---

### Section 17: Cropping (OPTIONAL but RECOMMENDED)

#### Cell 32: Crop Settings
```python
o_white_beam.crop_settings()
```

**What it does:**
- Interactive ROI selector on normalized image
- Define final field of view
- Different from pre-processing crop (applied after normalization)

#### Cell 33: Apply Crop
```python
o_white_beam.crop()
```

**What it does:**
- Crops `normalized_images` to selected region
- Updates image dimensions
- Reduces reconstruction computational cost

**Best Practice:**
- Include some background around sample
- Avoid cropping too tight (causes edge artifacts)
- Ensure sample stays within bounds during rotation

---

### Section 18: Sample Rotation (OPTIONAL)

#### Cell 34: Check if Rotation Needed
```python
o_white_beam.is_rotation_needed()
```

**What it does:**
- Displays current image orientation
- User decides if rotation needed

#### Cell 35: Rotation Settings
```python
o_white_beam.rotate_data_settings()
```

**What it does:**
- Select rotation angle (90°, 180°, 270°)
- Preview rotation result

#### Cell 36: Apply Rotation
```python
o_white_beam.apply_rotation()
```

**What it does:**
- Rotates all images in `normalized_images`
- Updates image dimensions if 90° or 270°

**Use Cases:**
- Sample axis not vertical in images
- Incorrect camera orientation during acquisition

#### Cell 37: Visualize After Rotation
```python
o_white_beam.visualize_after_rotation()
```

---

### Section 19: Log Conversion and Cleaning (MANDATORY for reconstruction)

#### Cell 38: Log Conversion
```python
o_white_beam.log_conversion_and_cleaning()
```

**What it does:**
- Converts to attenuation: `I_log = -log(I_normalized)`
- Removes negative values (unphysical)
- Creates `normalized_images_log`
- Essential for CT reconstruction

**Physics:**
Beer-Lambert law: `I = I₀ × exp(-μx)`
Log conversion: `-log(I/I₀) = μx`

**Also performs:**
- Outlier removal on log data
- Replaces NaN and Inf values
- Ensures data validity

#### Cell 39: Visualize After Log
```python
o_white_beam.visualize_images_after_log()
```

**What it does:**
- Compare normalized vs. log-converted images
- Verify no artifacts from log conversion

---

### Section 20: Strip Removal (OPTIONAL but RECOMMENDED)

Vertical or horizontal stripes are common artifacts in CT data from detector issues or beam fluctuations.

#### Cell 40: Select Test Range
```python
o_white_beam.select_range_of_data_to_test_stripes_removal()
```

**What it does:**
- Select subset of images for testing
- Faster to test on few images first

#### Cell 41: Select Algorithm
```python
o_white_beam.select_remove_strips_algorithms()
```

**What it does:**
- Choose stripe removal algorithm:
  - **Vo-all**: Wavelet-FFT based (algotom)
  - **Sarepy**: Stripe artifacts removal (algotom)
  - **FW**: Fourier-wavelet based (algotom)
  - **TI**: Titarenko algorithm (tomopy)
  - **SF**: Smoothing filter (tomopy)

#### Cell 42: Define Settings
```python
o_white_beam.define_settings()
```

**What it does:**
- Adjust algorithm-specific parameters
- Preview results on test images

#### Cell 43: Test on Selected Range
```python
o_white_beam.test_algorithms_on_selected_range_of_data()
```

**What it does:**
- Applies selected algorithm to test images
- Displays before/after comparison
- Shows parameter effects

#### Cell 44: When to Remove Strips
```python
o_white_beam.when_to_remove_strips()
```

**What it does:**
- Choose when to apply:
  - Before center of rotation
  - After center of rotation

#### Cell 45: Apply Strip Removal
```python
o_white_beam.remove_strips()
```

**What it does:**
- Applies stripe removal to all images
- Updates `normalized_images_log`
- Progress bar shows status

#### Cell 46: Display Results
```python
o_white_beam.display_removed_strips()
```

---

### Section 21: Tilt Correction (OPTIONAL but RECOMMENDED)

If rotation axis is tilted relative to detector, reconstruction will be blurred.

#### Cell 47: Select Sample ROI
```python
o_white_beam.select_sample_roi()
```

**What it does:**
- Select region containing sample
- Used for tilt calculation
- Exclude background for better results

#### Cell 48: Perform Tilt Correction
```python
o_white_beam.perform_tilt_correction()
```

**What it does:**
- Calculates tilt angle using neutompy
- Applies correction to all images
- Updates `normalized_images_log`

**Algorithm:**
- Finds rotation axis in 0° and 180° images
- Calculates tilt from axis positions
- Applies geometric correction

---

### Section 22: Center of Rotation (MANDATORY)

The center of rotation (COR) is critical for reconstruction quality.

#### Cell 49: COR Settings
```python
o_white_beam.center_of_rotation_settings()
```

**What it does:**
- Isolates 0°, 180°, and 360° images
- Configures COR search parameters:
  - Search range
  - Search step
  - Slice to use

#### Cell 50: Run COR Calculation
```python
o_white_beam.run_center_of_rotation()
```

**What it does:**
- Tests multiple COR values
- Reconstructs test slice for each value
- Calculates sharpness metric

#### Cell 51: Determine COR
```python
o_white_beam.determine_center_of_rotation()
```

**What it does:**
- Plots sharpness vs. COR value
- Identifies optimal COR (maximum sharpness)
- Stores result in `center_of_rotation`

#### Cell 52: Display COR Result
```python
o_white_beam.display_center_of_rotation()
```

**What it does:**
- Reconstructs with optimal COR
- Displays result
- Allows manual override if needed

**Methods:**
- Correlation-based (comparing 0° and 180°)
- Sharpness-based (maximum image sharpness)
- Manual adjustment

---

### Section 23: Sinogram Visualization (OPTIONAL)

#### Cell 53: Create Sinograms
```python
o_white_beam.create_sinograms()
```

**What it does:**
- Transposes data: `[angle, y, x]` → `[y, angle, x]`
- Each sinogram is horizontal slice through volume
- Creates `sinogram_normalized_images_log`

#### Cell 54: Visualize Sinograms
```python
o_white_beam.visualize_sinograms()
```

**What it does:**
- Browse through sinograms
- Verify data quality
- Identify remaining artifacts

**Sinogram Quality Checks:**
- Smooth sinusoidal patterns (good)
- Breaks or jumps (missing angles)
- Horizontal lines (stripes)
- Vertical lines (ring artifacts)

---

### Section 24: Test Reconstruction (OPTIONAL but RECOMMENDED)

Quick reconstruction test before processing entire dataset.

#### Cell 55: Select Slices to Test
```python
o_white_beam.select_slices_to_use_to_test_reconstruction()
```

**What it does:**
- Select 2-5 representative slices
- Choose slices with different features

#### Cell 56: Run Test Reconstruction
```python
o_white_beam.run_reconstruction_of_slices_to_test()
```

**What it does:**
- Reconstructs selected slices using gridrec (fast)
- Displays results
- Verifies COR and preprocessing quality

**Check for:**
- Sharp features (good COR)
- No doubling (correct COR)
- Good contrast (proper preprocessing)
- Minimal artifacts

---

### Section 25: Select Reconstruction Method (MANDATORY)

#### Cell 57: Select Method
```python
o_white_beam.select_reconstruction_method()
```

**What it does:**
- Choose reconstruction algorithm(s):
  - **FBP (Filtered Back Projection)**: Fast, analytical
  - **gridrec**: Fast, Fourier-based
  - **SVMBIR**: Iterative, best quality, slow
  - **MBIRJAX**: GPU-accelerated iterative

**Algorithm Comparison:**

| Algorithm | Speed | Quality | Memory | Use Case |
|-----------|-------|---------|--------|----------|
| FBP | Fast | Good | Low | Standard reconstructions |
| gridrec | Very Fast | Good | Low | Quick tests |
| SVMBIR | Slow | Excellent | High | Publication-quality |
| MBIRJAX | Medium | Excellent | GPU | Large datasets |

---

### Section 26: Reconstruction Settings (MANDATORY)

#### Cell 58: Configure Reconstruction
```python
o_white_beam.reconstruction_settings()
```

**What it does:**
- Configure algorithm-specific parameters

**FBP/gridrec Settings:**
- Filter type (shepp, ramlak, butterworth, etc.)
- Filter cutoff frequency

**SVMBIR/MBIRJAX Settings:**
- Number of iterations
- Regularization parameters:
  - `sharpness`: Controls edge preservation
  - `T`: Temperature parameter
- Pixel size
- Distance units

**Best Practice:**
- Start with default parameters
- Adjust based on test reconstruction results
- Higher sharpness = sharper edges but more noise
- More iterations = better quality but slower

---

### Section 27: Export Configuration and Data (MANDATORY)

#### Cell 59: Select Export Folder
```python
o_white_beam.select_export_extra_files()
```

**What it does:**
- Select output directory for:
  - Configuration JSON file
  - Log file
  - Preprocessed data

#### Cell 60: Export Files
```python
o_white_beam.export_extra_files(prefix='step1')
```

**What it does:**
- Exports configuration JSON (`step1_####.json`)
- Exports processing log
- Exports preprocessed projections as TIFF stack
- Exports angle list
- Exports metadata

**Configuration File Contents:**
```json
{
  "data_folder": "/path/to/data",
  "angles": [0.0, 0.5, 1.0, ...],
  "center_of_rotation": 1024.5,
  "reconstruction_algorithm": "svmbir",
  "reconstruction_parameters": {
    "sharpness": 0.0,
    "iterations": 100,
    ...
  },
  "image_dimensions": [2048, 2048],
  "preprocessing": {
    "crop": {...},
    "rotation": 0,
    "normalization": true,
    ...
  }
}
```

**Output Files:**
```
output_folder/
├── step1_config_20250611_143022.json    # Configuration
├── step1_log_20250611_143022.log        # Processing log  
├── projections/
│   ├── projection_000.tif               # Preprocessed images
│   ├── projection_001.tif
│   └── ...
└── angles.txt                            # Angle list
```

These files are used by Step 2 (slicing) and Step 3 (reconstruction).

---

## API Reference

### Main Class Methods

#### Initialization
```python
Step1PrepareCcdImages(system=None)
```
Initialize the workflow with system configuration.

---

#### Data Selection Methods

**select_top_sample_folder()**
```python
o_white_beam.select_top_sample_folder()
```
Opens file browser to select folder containing sample projections.

**select_ob_images()**
```python
o_white_beam.select_ob_images()
```
Opens file browser to select open beam images.

**select_dc_images()**
```python
o_white_beam.select_dc_images()
```
Opens file browser to select dark current images.

---

#### Angle Retrieval Methods

**how_to_retrieve_angle_value()**
```python
o_white_beam.how_to_retrieve_angle_value()
```
Presents options for angle retrieval method.

**retrieve_angle_value()**
```python
o_white_beam.retrieve_angle_value()
```
Retrieves angles using selected method.

**testing_angle_values()**
```python
o_white_beam.testing_angle_values()
```
Displays and validates retrieved angles.

---

#### Data Loading

**load_data()**
```python
o_white_beam.load_data()
```
Loads all selected images into memory.

- **Updates**: `master_3d_data_array`, `final_list_of_angles`
- **Returns**: None

---

#### Visualization Methods

**visualize_raw_data()**
```python
o_white_beam.visualize_raw_data()
```
Interactive visualization of raw projections.

**visualize_cleaned_data()**
```python
o_white_beam.visualize_cleaned_data()
```
Visualize images after outlier removal.

**visualize_normalization()**
```python
o_white_beam.visualize_normalization()
```
Compare cleaned vs. normalized images.

---

#### Image Processing Methods

**clean_images_settings()**
```python
o_white_beam.clean_images_settings()
```
Configure outlier removal parameters.

**clean_images()**
```python
o_white_beam.clean_images()
```
Apply outlier removal to all images.

- **Updates**: `master_3d_data_array`

**normalization()**
```python
o_white_beam.normalization()
```
Perform OB/DC normalization.

- **Creates**: `normalized_images`

**crop()**
```python
o_white_beam.crop()
```
Crop images to ROI.

- **Updates**: `normalized_images`, `crop_region`

**apply_rotation()**
```python
o_white_beam.apply_rotation()
```
Rotate all images by specified angle.

- **Updates**: `normalized_images`

**log_conversion_and_cleaning()**
```python
o_white_beam.log_conversion_and_cleaning()
```
Convert to attenuation and clean.

- **Creates**: `normalized_images_log`

**remove_strips()**
```python
o_white_beam.remove_strips()
```
Apply stripe removal algorithm.

- **Updates**: `normalized_images_log`

**perform_tilt_correction()**
```python
o_white_beam.perform_tilt_correction()
```
Calculate and apply tilt correction.

- **Updates**: `normalized_images_log`

---

#### Center of Rotation Methods

**center_of_rotation_settings()**
```python
o_white_beam.center_of_rotation_settings()
```
Configure COR calculation parameters.

**run_center_of_rotation()**
```python
o_white_beam.run_center_of_rotation()
```
Calculate optimal center of rotation.

**determine_center_of_rotation()**
```python
o_white_beam.determine_center_of_rotation()
```
Display and finalize COR value.

- **Updates**: `center_of_rotation`

---

#### Reconstruction Methods

**select_reconstruction_method()**
```python
o_white_beam.select_reconstruction_method()
```
Select reconstruction algorithm(s).

- **Updates**: `configuration.reconstruction_algorithm`

**reconstruction_settings()**
```python
o_white_beam.reconstruction_settings()
```
Configure reconstruction parameters.

**run_reconstruction_of_slices_to_test()**
```python
o_white_beam.run_reconstruction_of_slices_to_test()
```
Quick reconstruction of test slices.

---

#### Export Methods

**export_extra_files(prefix="")**
```python
o_white_beam.export_extra_files(prefix='step1')
```
Export configuration, log, and preprocessed data.

- **Parameters**:
  - `prefix` (str): Prefix for output filenames

---

## Data Structures

### 3D Array Shapes

Throughout the workflow, data arrays have specific shapes:

```python
# Raw projections
master_3d_data_array[DataType.sample].shape = (n_angles, height, width)

# After processing
normalized_images.shape = (n_angles, height_cropped, width_cropped)
normalized_images_log.shape = (n_angles, height_cropped, width_cropped)

# Sinograms
sinogram_normalized_images_log.shape = (height, n_angles, width)
```

### Configuration Object

```python
configuration = {
    'reconstruction_algorithm': [ReconstructionAlgorithm.svmbir],
    'angles': array([0.0, 0.5, 1.0, ...]),
    'center_of_rotation': 1024.5,
    'data_folder': '/path/to/data',
    'crop_region': {'left': 0, 'right': 2048, 'top': 0, 'bottom': 2048},
    'preprocessing': {
        'cleaning': True,
        'normalization': True,
        'stripe_removal': True,
        'tilt_correction': True,
    },
    'reconstruction_parameters': {
        'svmbir': {
            'sharpness': 0.0,
            'T': 1.0,
            'iterations': 100,
        }
    }
}
```

---

## Configuration

### Algorithm Parameters

#### SVMBIR Parameters

```python
{
    'sharpness': 0.0,      # 0.0-2.0, higher = sharper edges
    'T': 1.0,              # Temperature, typically 1.0-2.0
    'p': 1.2,              # Norm parameter, 1.0-2.0
    'q': 2.0,              # Norm parameter
    'iterations': 100,     # Number of iterations
    'positivity': True,    # Enforce positive values
}
```

#### FBP Parameters

```python
{
    'filter': 'shepp',     # Filter type: shepp, ramlak, butterworth, etc.
    'cutoff': 1.0,         # Cutoff frequency (0.0-1.0)
}
```

### Stripe Removal Algorithms

#### Vo-all (algotom)
```python
{
    'snr': 3.0,           # Signal-to-noise ratio
    'la_size': 61,        # Vertical local averaging window
    'sm_size': 21,        # Horizontal smoothing window
}
```

#### Sarepy (algotom)
```python
{
    'snr': 3.0,           # Signal-to-noise ratio
}
```

---

## Troubleshooting

### Common Issues

#### 1. Memory Error During Loading

**Error**: `MemoryError: Unable to allocate array`

**Solutions**:
- Load only fraction of data
- Use rebinning to reduce image size
- Process in batches
- Increase system RAM

#### 2. Angles Not Found in Metadata

**Error**: `MetadataError: Rotation angle not found in NeXus`

**Solutions**:
- Use filename parsing method
- Use manual angle entry
- Check NeXus file structure
- Verify metadata field names

#### 3. Poor Reconstruction Quality

**Symptoms**: Blurry, doubled features, ring artifacts

**Solutions**:
- **Blurry**: Recheck center of rotation
- **Doubled**: Wrong COR value
- **Rings**: Apply stripe removal
- **Low contrast**: Adjust reconstruction parameters

#### 4. Normalization Issues

**Symptoms**: Bright/dark bands, uneven illumination

**Solutions**:
- Check OB image quality
- Ensure OB images taken without sample
- Use more OB images for averaging
- Verify DC images are dark (shutter closed)

#### 5. Stripe Artifacts

**Symptoms**: Vertical lines in sinograms, rings in reconstruction

**Solutions**:
- Apply stripe removal algorithms
- Try different algorithms (Vo-all usually works well)
- Adjust algorithm parameters
- Check for dead detector pixels

---

## Best Practices

### Data Quality

1. **Acquire sufficient OB images**: 10-20 images minimum
2. **Regular DC acquisitions**: Before and after sample scans
3. **Consistent acquisition parameters**: Keep exposure time, binning constant
4. **Overlap angles**: Acquire 360° + few degrees for consistency check
5. **Angular sampling**: Follow Nyquist criterion (~500-1000 angles for typical samples)

### Processing Workflow

1. **Always visualize raw data** before processing
2. **Test on subset** of data before full processing
3. **Save intermediate results** (normalized images, etc.)
4. **Document parameters** used for reproducibility
5. **Quick reconstruction test** before full reconstruction

### Parameter Selection

1. **Start with defaults** for most parameters
2. **Outlier removal**: Mild settings first, increase if needed
3. **Stripe removal**: Test on few images first
4. **COR**: Use automated calculation, verify visually
5. **Reconstruction**: Start with FBP, use SVMBIR for final results

### Memory Management

1. **Monitor memory usage** during loading
2. **Use rebinning** if memory constrained
3. **Close visualizations** when not needed
4. **Clear kernel** between runs if memory issues

### File Organization

1. **Consistent directory structure**
2. **Meaningful filenames** with dates/sample IDs
3. **Keep raw data separate** from processed
4. **Version control** for configuration files
5. **Document** experiment conditions

---

## Workflow Diagram

```
┌─────────────────────────────────────────────┐
│  1. Initialize & Select Data                │
│     - Sample folder                         │
│     - OB images (optional)                  │
│     - DC images (optional)                  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  2. Retrieve Angles                         │
│     - From metadata / filename / manual     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  3. Load Data                               │
│     - Load all images into memory           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  4. Pre-processing (Optional)               │
│     - Exclude bad images                    │
│     - Pre-crop                              │
│     - Clean outliers                        │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  5. Normalization (if OB available)         │
│     - Combine OB/DC                         │
│     - Normalize: (S-DC)/(OB-DC)             │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  6. Geometric Corrections                   │
│     - Rebin (optional)                      │
│     - Crop                                  │
│     - Rotate (optional)                     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  7. Log Conversion                          │
│     - Convert to attenuation                │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  8. Artifact Removal                        │
│     - Remove stripes                        │
│     - Tilt correction                       │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  9. Center of Rotation                      │
│     - Calculate COR                         │
│     - Test reconstruction                   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  10. Configure Reconstruction               │
│      - Select algorithm                     │
│      - Set parameters                       │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  11. Export                                 │
│      - Configuration JSON                   │
│      - Preprocessed data                    │
│      - Log file                             │
└─────────────────────────────────────────────┘
```

---

## Next Steps

After completing Step 1, proceed to:

- **Step 2**: Slice selection (`step2_slice_CCD_or_TimePix_images.ipynb`)
  - Select specific slices to reconstruct
  - Reduce computational cost for large datasets

- **Step 3**: Reconstruction (`step3_reconstruction_CCD_or_TimePix_images.py`)
  - Full 3D reconstruction
  - CLI or notebook interface

- **Step 4**: Visualization (`step4_visualization_CCD_images.ipynb`)
  - 3D volume rendering
  - Slice-by-slice analysis
  - Quantitative measurements

---

## References

### Algorithms

- **FBP**: Kak, A. C., & Slaney, M. (1988). Principles of Computerized Tomographic Imaging
- **gridrec**: Dowd et al. (1999). "Developments in Synchrotron X-Ray Computed Microtomography"
- **SVMBIR**: Venkatakrishnan et al. (2013). "Plug-and-Play priors for model based reconstruction"
- **Stripe removal (Vo)**: Vo et al. (2018). "Superior techniques for eliminating ring artifacts in X-ray micro-tomography"

### Software

- **iMars3D**: https://github.com/ornlneutronimaging/iMars3D
- **TomoPy**: https://github.com/tomopy/tomopy
- **Algotom**: https://github.com/algotom/algotom
- **SVMBIR**: https://github.com/cabouman/svmbir

---

## Glossary

- **CCD**: Charge-Coupled Device detector
- **COR**: Center of Rotation
- **CT**: Computed Tomography
- **DC**: Dark Current (background signal with closed shutter)
- **FBP**: Filtered Back Projection
- **IPTS**: Integrated Proposal Tracking System (experiment ID)
- **NeXus**: HDF5-based data format for neutron/X-ray facilities
- **OB**: Open Beam (reference measurement without sample)
- **ROI**: Region of Interest
- **Sinogram**: 2D image showing projections vs. angle for one slice
- **SVMBIR**: Sparse-View Model-Based Iterative Reconstruction
- **TOF**: Time-of-Flight
- **White Beam**: Broad spectrum (non-monochromatic) beam

---

## Support

For issues or questions:
- Check the troubleshooting section
- Review visualization outputs for data quality issues
- Consult tomography reconstruction literature
- Contact beamline scientists for instrument-specific questions

---

*Document Version: 1.0*  
*Last Updated: 2025-11-11*  
*Author: Auto-generated documentation for all_ct_reconstruction workflow*
