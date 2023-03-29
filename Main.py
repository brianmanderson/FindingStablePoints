import SimpleITK as sitk
import numpy as np
from PlotScrollNumpyArrays import plot_scroll_Image
import os


dose_file = r'dose.nii.gz'
reader = sitk.ImageFileReader()
reader.SetFileName(dose_file)
reader.ReadImageInformation()
dose = reader.Execute()
dose_np = sitk.GetArrayFromImage(dose)
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
    image_cube = masked_np[z_start:z_stop, r_start:r_stop, c_start:c_stop]