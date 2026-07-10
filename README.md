# FindingStablePoints

Finds "stable" measurement points in a 3D dose distribution for patient-specific QA:
locations inside each high-dose region where the dose gradient is lowest over the physical
footprint of the detector, so a point-dose measurement there is least sensitive to
positioning error. Single-script utility; not actively maintained (last updated 2023).

## How it works

`Main.py` provides one routine, `return_physical_location(dose_file, limit=0.9, camera_dimensions=(3, 5, 3))`:

1. Loads a NIfTI dose file with SimpleITK.
2. Thresholds the dose at `limit` x the maximum dose and labels each connected high-dose region (one per treatment site).
3. Within each region's bounding box, computes the dose gradient magnitude and smooths it with an averaging kernel sized to `camera_dimensions` (detector dimensions in mm, converted to voxels), so the whole detector footprint sits in a low-gradient area.
4. Takes the center of the minimum-gradient area, converts it to a physical (patient-space) coordinate, and prints it for each site.

## Usage

```python
from Main import return_physical_location

return_physical_location(r'dose.nii.gz', limit=0.9)
# Identified position for one site: (-5.5615081787109375, -1.0670852661132812, -1.25)
```

Running `python Main.py` runs the bundled example on the sample `dose.nii.gz` included
in the repo (limit=0.85, 5 mm kernel). Only NIfTI dose input is supported; the DICOM
dose loading mentioned in the original README was never added.

## Requirements

Python with SimpleITK and NumPy (the script also imports SciPy, Matplotlib, and
PlotScrollNumpyArrays, but the core routine uses only SimpleITK and NumPy).
