import os
import numpy as np
from pathlib import Path
from pydicom import dcmread

# Loading one folder -------------------------------------------------------------------------------
def load_dicom_folder(folder_path: Path):
    dicoms = []

    for img in folder_path.iterdir():
        print(f"iterdir:{img}")
        if img.is_file():
            dicoms.append(str(img))

    return dicoms

# Loading all the patients -------------------------------------------------------------------------
def load_all_dicom(base_path: str):
    """
    dicom_type could be p:patient, l:labelled, or m:masks
    """
    patient_extension_path = Path("PATIENT_DICOM") / "PATIENT_DICOM"
    labelled_extension_path = Path("LABELLED_DICOM") / "LABELLED_DICOM"
    masks_extension_path = Path("MASKS_DICOM") / "MASKS_DICOM" / "liver"

    all_patient = []
    all_labelled = []
    all_masks = []

    for current_path in base_path.iterdir(): # patients are not in order, problem???????? sort????
        print(f"current:{current_path}")
        if current_path.is_dir():
            patient_path = current_path / patient_extension_path
            labelled_path = current_path / labelled_extension_path
            masks_path = current_path / masks_extension_path

            # the rest is the same as load_patient_dicom, we don't use the load_dicom file because
            # appending is faster and more efficient than concatenating
            for img in patient_path.iterdir(): # same number of elements in each of the three folders, one loop is ok
                print(f"img:{img}")
                if img.is_file():
                    all_patient.append(str(img))
                    image_name = os.path.splitext(img)[0]
                    all_labelled.append(str(labelled_path / image_name))
                    all_masks.append(str(masks_path / image_name))
                    

    return all_patient, all_labelled, all_masks

if __name__ == "__main__":
    base_path = Path("C:/Users/HP/Desktop/PIMA/3Dircadb1")    
    patients, labelled, masks = load_all_dicom(base_path, dicom_type="m")
    print(labelled)
    print(len(patients), len(np.unique(patients)))
    assert len(np.unique(patients)) == len(np.unique(labelled))
    assert len(labelled) == len(np.unique(masks))