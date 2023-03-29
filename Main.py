import SimpleITK as sitk
import numpy as np
from matplotlib.pyplot import plot
from PlotScrollNumpyArrays import plot_scroll_Image
import os

dose_handle: sitk.Image
dose_file = r'dose.nii.gz'
reader = sitk.ImageFileReader()
reader.SetFileName(dose_file)
reader.ReadImageInformation()
dose_handle = reader.Execute()
dose_np = sitk.GetArrayFromImage(dose_handle)
xxx = 1
limit = 0.9
max_dose = np.max(dose_np)
Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
Connected_Threshold = sitk.ConnectedThresholdImageFilter()
Connected_Threshold.SetLower(1)
Connected_Threshold.SetUpper(2)
truth_stats = sitk.LabelShapeStatisticsImageFilter()
'''
Next, identify each independent segmentation in both
'''
masked_np = (dose_np >= limit*max_dose).astype('int')
base_mask = sitk.GetImageFromArray(masked_np)
connected_image_handle = Connected_Component_Filter.Execute(base_mask)

RelabelComponentFilter = sitk.RelabelComponentImageFilter()
connected_image = RelabelComponentFilter.Execute(connected_image_handle)
truth_stats.Execute(connected_image)
bounding_boxes = np.asarray([truth_stats.GetBoundingBox(l) for l in truth_stats.GetLabels()])
for index in range(bounding_boxes.shape[0]):
    bounding_box = bounding_boxes[index]
    c_start, r_start, z_start, _, _, _ = bounding_box
    c_stop, r_stop, z_stop = c_start + bounding_box[3], r_start + bounding_box[4], z_start + bounding_box[5]
    dose_cube = dose_np[z_start:z_stop, r_start:r_stop, c_start:c_stop]
    gradient_np = np.abs(np.gradient(dose_cube, 2))
    super_imposed_gradient_np = np.sum(gradient_np, axis=0)
    min_gradient = np.min(super_imposed_gradient_np)
    min_location = np.where(super_imposed_gradient_np <= min_gradient*1.1)
    min_z = int(z_start + min_location[0][0])
    min_row = int(r_start + min_location[1][0])
    min_col = int(c_start + min_location[2][0])
    physical_location = dose_handle.TransformContinuousIndexToPhysicalPoint((min_col, min_row, min_z))
    xx = 1