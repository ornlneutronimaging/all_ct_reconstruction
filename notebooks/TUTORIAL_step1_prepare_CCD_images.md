# Tutorial: CCD Image Preparation for CT Reconstruction

This tutorial provides a comprehensive guide for running the `step1_prepare_CCD_images.ipynb` notebook, which is the first step in the neutron CT reconstruction pipeline for CCD detector data.

## Overview

The CCD image preparation notebook processes raw projection images from CCD detectors and prepares them for CT reconstruction. This includes data loading, normalization, cleaning, preprocessing, and parameter optimization.

## Prerequisites

### System Requirements
- Access to SNS/ORNL neutron imaging beamline data
- Python environment with required packages (see `requirements.txt`)
- Sufficient memory for large image datasets
- IPTS (Integrated Program Tracking System) access

### Data Requirements
- **Sample projections**: Folder containing projection images (one per rotation angle)
- **Open beam images** (optional): For normalization
- **Dark current images** (optional): For background subtraction
- **Metadata**: Angle information (from NeXus files or manual input)

## Step-by-Step Workflow

### 1. Initial Setup

```python
import warnings
warnings.filterwarnings('ignore')

from __code.step1_prepare_ccd_images import Step1PrepareCcdImages
from __code import system

system.System.select_working_dir(ipts="IPTS-33767", instrument="CG1D")
from __code.__all import custom_style
custom_style.style()

Step1PrepareCcdImages.legend()
```

**What this does:**
- Imports required modules
- Sets up working directory for your IPTS and instrument
- Applies notebook styling
- Displays the workflow legend

**Action required:** Update the IPTS number to match your experiment.

### 2. Data Input

#### 2.1 Select Sample Folder
```python
o_white_beam = Step1PrepareCcdImages(system=system)
o_white_beam.select_top_sample_folder()
```

**Instructions:**
- Select the folder containing your projection images
- Each file should represent one rotation angle
- Ensure files are properly named and organized

#### 2.2 Select Open Beam Images (Optional)
```python
o_white_beam.select_ob_images()
```

**When to use:**
- If your data is not already normalized
- For improving image quality through flat-field correction

**Instructions:**
- Select individual open beam (flat field) images
- These images should be taken without the sample in the beam

#### 2.3 Select Dark Current Images (Optional)
```python
o_white_beam.select_dc_images()
```

**When to use:**
- If you selected open beam images
- For background noise subtraction

**Instructions:**
- Select dark current images (taken with beam off)
- Will be ignored if no open beam images were selected

### 3. Angle Configuration

#### 3.1 Choose Angle Retrieval Method
```python
o_white_beam.how_to_retrieve_angle_value()
```

**Options:**
- **From NeXus files**: Automatic extraction from metadata
- **Manual input**: Define angle range and increment
- **From filename**: Extract from image filenames

#### 3.2 Retrieve Angles
```python
o_white_beam.retrieve_angle_value()
```

**What this does:**
- Extracts or defines rotation angles for each projection
- Critical for proper reconstruction geometry

#### 3.3 Test Angle Values
```python
o_white_beam.testing_angle_values()
```

**Purpose:**
- Verify angle values are correct
- Check for missing angles or duplicates

### 4. Data Subset Selection

#### 4.1 Choose Data Amount
```python
o_white_beam.use_all_or_fraction()
```

**Options:**
- **All data**: Use complete dataset
- **Fraction**: Use subset for testing or memory constraints

#### 4.2 Select Percentage (if applicable)
```python
o_white_beam.select_percentage_of_data_to_use()
```

**When to use:**
- Large datasets that exceed memory limits
- Quick testing of processing pipeline

### 5. Data Loading

```python
o_white_beam.load_data()
```

**What this does:**
- Loads all projection images into memory
- Converts to white beam mode (sums time-of-flight channels)
- Sorts images by increasing angle
- Creates master data arrays

**Note:** This step may take significant time for large datasets.

### 6. Data Visualization (Optional)

#### 6.1 Choose Visualization Mode
```python
o_white_beam.how_to_visualize()
```

**Options:**
- **All images**: Complete dataset overview (slow for large data)
- **Visual verification**: Quick check of raw, OB, and DC images

#### 6.2 Visualize Raw Data
```python
o_white_beam.visualize_raw_data()
```

**Purpose:**
- Quality check of loaded data
- Identify potential issues early

### 7. Image Exclusion (Optional)

#### 7.1 Select Exclusion Mode
```python
o_white_beam.selection_mode()
```

**When to use:**
- Remove problematic projections
- Exclude damaged or corrupted images

#### 7.2 Process Exclusion
```python
o_white_beam.process_exclusion_mode()
o_white_beam.exclude_this_list_of_images()
```

### 8. Preprocessing Steps

#### 8.1 Crop Raw Data (Optional)
```python
o_white_beam.pre_processing_crop_settings()
o_white_beam.pre_processing_crop()
```

**Purpose:**
- Reduce data size
- Focus on region of interest
- Improve processing speed

#### 8.2 Remove Outliers (Optional)
```python
o_white_beam.clean_images_settings()
o_white_beam.clean_images_setup()  # Only for histogram method
o_white_beam.clean_images()
```

**Available algorithms:**
- **Histogram method**: Removes dead pixels and abnormal high counts
- **TomoPy remove_outlier**: Removes bright spots
- **SciPy gamma_filter**: Statistical outlier removal

#### 8.3 Visualize Cleaned Data (Optional)
```python
o_white_beam.how_to_visualize_after_cleaning()
o_white_beam.visualize_cleaned_data()
```

#### 8.4 Rebin Pixels (Optional)
```python
o_white_beam.rebin_settings()
o_white_beam.rebin_before_normalization()
```

**Purpose:**
- Combine adjacent pixels to reduce noise
- Decrease data size for faster processing

### 9. Normalization

#### 9.1 Normalization Settings
```python
o_white_beam.normalization_settings()
```

**Options:**
- **Use sample background ROI**: Scale to transmission value of 1
- **Use background ROI**: Match open beam background regions

#### 9.2 Select ROI (if applicable)
```python
o_white_beam.normalization_select_roi()
```

**Important:**
- Select region **outside** your sample
- Ensure consistent background across all projections

#### 9.3 Perform Normalization
```python
o_white_beam.normalization()
```

**What this does:**
- Applies flat-field correction
- Background subtraction
- Creates normalized image stack

### 10. Post-Normalization Processing

#### 10.1 Visualize Normalized Data
```python
o_white_beam.visualization_normalization_settings()
o_white_beam.visualize_normalization()
```

#### 10.2 Export Normalized Images (Optional)
```python
o_white_beam.select_export_normalized_folder()
o_white_beam.export_normalized_images()
```

#### 10.3 Additional Rebinning (Optional)
```python
o_white_beam.rebin_settings()
o_white_beam.rebin_after_normalization()
```

#### 10.4 Final Cropping (Optional)
```python
o_white_beam.crop_settings()
o_white_beam.crop()
```

### 11. Geometric Corrections

#### 11.1 Rotation Correction
```python
o_white_beam.is_rotation_needed()
o_white_beam.rotate_data_settings()
o_white_beam.apply_rotation()
```

**Critical requirement:**
- Rotation axis must be **vertical** for reconstruction
- Rotate data if axis is not aligned properly

### 12. Logarithmic Conversion

```python
o_white_beam.log_conversion_and_cleaning()
```

**Purpose:**
- Convert transmission data to attenuation
- Required for reconstruction algorithms
- Applies -ln(I/I₀) transformation

### 13. Advanced Processing

#### 13.1 Visualize Sinograms
```python
o_white_beam.visualize_sinograms()
```

**Purpose:**
- Check data quality in sinogram space
- Identify systematic issues

#### 13.2 Stripe Removal (Optional)

**Step-by-step approach:**

1. **Select test range:**
```python
o_white_beam.select_range_of_data_to_test_stripes_removal()
```

2. **Choose algorithms:**
```python
o_white_beam.select_remove_strips_algorithms()
```

3. **Define settings:**
```python
o_white_beam.define_settings()
```

4. **Test algorithms:**
```python
o_white_beam.test_algorithms_on_selected_range_of_data()
```

5. **Apply to full dataset:**
```python
o_white_beam.when_to_remove_strips()
o_white_beam.remove_strips()
```

**Options for timing:**
- **In notebook**: Process now (may be slow)
- **During reconstruction**: Process in background (recommended)

#### 13.3 Tilt Correction (Optional)
```python
o_white_beam.select_sample_roi()
o_white_beam.perform_tilt_correction()
```

**Purpose:**
- Corrects sample tilt using 0° and 180° projections
- Improves reconstruction quality

### 14. Center of Rotation

#### 14.1 Settings and Calculation
```python
o_white_beam.center_of_rotation_settings()
o_white_beam.run_center_of_rotation()
o_white_beam.determine_center_of_rotation()
```

**Methods:**
- **Automatic**: Uses algotom library
- **Manual**: Interactive determination

**Importance:**
- Critical parameter for reconstruction quality
- Incorrect center causes blurring and artifacts

### 15. Test Reconstruction

```python
o_white_beam.select_slices_to_use_to_test_reconstruction()
o_white_beam.run_reconstruction_of_slices_to_test()
```

**Purpose:**
- Validate all processing steps
- Test reconstruction parameters
- Quick quality check before full reconstruction

### 16. Reconstruction Setup

#### 16.1 Select Method
```python
o_white_beam.select_reconstruction_method()
```

**Available methods:**
- Various reconstruction algorithms
- Choose based on your requirements

#### 16.2 Configure Parameters
```python
o_white_beam.reconstruction_settings()
```

**Tip:** Default values are good for novice users.

### 17. Export and Finalize

```python
o_white_beam.select_export_extra_files()
o_white_beam.export_extra_files(prefix='step1')
```

**Generated files:**
- **Config file** (`step1_*.json`): For use in reconstruction
- **Log file**: Processing record
- **Preprocessed images**: Ready for reconstruction

## Best Practices

### Memory Management
- Use data fractions for large datasets during testing
- Close visualization windows when not needed
- Monitor memory usage throughout processing

### Quality Control
- Always visualize data after major processing steps
- Test reconstruction on small subset before full processing
- Keep original data backups

### Parameter Optimization
- Test different cleaning algorithms on small datasets
- Optimize center of rotation carefully
- Use appropriate binning for your resolution requirements

### Troubleshooting

#### Common Issues

1. **Memory errors:**
   - Reduce data fraction
   - Increase system RAM
   - Process in smaller batches

2. **Incorrect angles:**
   - Verify NeXus file metadata
   - Check filename patterns
   - Use manual angle definition if needed

3. **Poor normalization:**
   - Verify open beam and dark current images
   - Check ROI selection
   - Ensure consistent experimental conditions

4. **Reconstruction artifacts:**
   - Verify center of rotation
   - Check for remaining stripes
   - Validate tilt correction

#### Performance Tips

- Use background processing for stripe removal on large datasets
- Export intermediate results for checkpoint recovery
- Process test subsets before full datasets

## Next Steps

After successful completion:
1. Use generated config file in step 2 (slicing)
2. Proceed to reconstruction with prepared images
3. Apply lessons learned to optimize future processing

## Support

For additional help:
- Check log files for detailed processing information
- Review visualization outputs for quality assessment
- Consult beamline scientists for experiment-specific guidance