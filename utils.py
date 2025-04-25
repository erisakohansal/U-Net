import os
import numpy as np
import zipfile
from pydicom import dcmread
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path #https://docs.python.org/3/library/pathlib.html
import shutil

LIVER_INDICES = {1:  (27 , 124),
                 2:  (44 , 157),
                 3:  (61 , 182),
                 4:  (19 , 85 ),
                 5:  (19 , 137),
                 6:  (37 , 132),
                 7:  (48 , 145),
                 8:  (6  , 121),
                 9:  (11 , 99 ),
                 10: (22 , 120),
                 11: (33 , 128),
                 12: (9  , 247),
                 13: (26 , 115),
                 14: (4  , 107),
                 15: (6  , 120),
                 16: (39 , 151),
                 17: (1  , 113),
                 18: (12 , 70 ),
                 19: (28 , 68 ),
                 20: (126, 210),
            }

# Folder related functions -----------------------------------------------------------------------
def unzip(zip_path, extract_path):
    """
    Extracts the compressed zip folder zip_path
    in the extract_path. If the extract_path folder
    doesn't exist already, it is created before the 
    extraction.
    """
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"{zip_path} is extracted to: {extract_path}")
    return

def dicom_to_nifti(): pass

# DICOM preprocessing -----------------------------------------------------------------------------------------------------
def dicom_preprocess(path, min_hu=-100, max_hu=400):
    """
    Performs preprocessing on the DICOM image located at path.
    """
    dcm_file = dcmread(path)
    
    pixel_data = dcm_file.pixel_array.astype(np.float32)

    # HU conversion
    slope = getattr(dcm_file, "RescaleSlope", 1.0)
    intercept = getattr(dcm_file, "RescaleIntercept", 0.0)
    pixel_hu = pixel_data * slope + intercept

    # HU clipping, d'après Halder et al. il faut qu'il soit entre [-200,200] pour la tâche de la segmentation du foie
    pixel_clipped = np.clip(pixel_hu, min_hu, max_hu)
    return pixel_clipped

def nifti_preprocess(path): pass

# Load all the liver present dicom pathes from the dataset -------------------------------------------
def load_all_liver_appearances(base_path:Path, num_patients=[]):
    global LIVER_INDICES
    images, masks, patient_id = [], [], []

    to_load = LIVER_INDICES.keys() if num_patients == [] else num_patients
    
    for num_patient in to_load:
        first, last = LIVER_INDICES[num_patient]
        for ind in range(first, last+1):
            images.append(base_path/f"3Dircadb1/3Dircadb1.{num_patient}/PATIENT_DICOM/PATIENT_DICOM/image_{ind}")
            masks.append(base_path/f"3Dircadb1/3Dircadb1.{num_patient}/MASKS_DICOM/MASKS_DICOM/liver/image_{ind}")
            patient_id.append(num_patient)
            
    return images, masks, patient_id
