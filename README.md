# FindingStablePoints
A code designed to find stable points for patient specific QA

Below is an example which comes from a nifti dose file, will add ability to simply load from a DICOM dose file shortly

    
    def run():
        dose_file = r'dose.nii.gz'
        return_physical_location(dose_file, limit=0.9)
    Identified position for one site: (-5.5615081787109375, -1.0670852661132812, -1.25)
