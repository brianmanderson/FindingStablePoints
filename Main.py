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