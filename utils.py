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

def train_test_division(base_path:Path, jpg=False, verbose=False, train_patients=list(range(1,19))):
    os.makedirs(base_path/"Data/Train/Images", exist_ok=True)  
    os.makedirs(base_path/"Data/Test/Images", exist_ok=True) 
    os.makedirs(base_path/"Data/Train/Masks", exist_ok=True)  
    os.makedirs(base_path/"Data/Test/Masks", exist_ok=True) 

    for num_patient in range(1, 21):

        if num_patient in train_patients:
            current_folder = "Train"
        else:
            current_folder = "Test"

        for filename in os.listdir(base_path/f"3Dircadb1/3Dircadb1.{num_patient}/PATIENT_DICOM/PATIENT_DICOM"):
            elems = filename.split("_", 1)
            num_slice = elems[1]

            dicom_to_png_jpeg(dicom_dir=base_path/f"3Dircadb1/3Dircadb1.{num_patient}/PATIENT_DICOM/PATIENT_DICOM/image_{num_slice}", filename=f"image_{num_patient}_{num_slice}", save_path=base_path/"Data"/current_folder/"Images", jpg=jpg, verbose=verbose, preprocess=True)
            dicom_to_png_jpeg(dicom_dir=base_path/f"3Dircadb1/3Dircadb1.{num_patient}/MASKS_DICOM/MASKS_DICOM/liver/image_{num_slice}", filename=f"image_{num_patient}_{num_slice}", save_path=base_path/"Data"/current_folder/"Masks", jpg=jpg, verbose=verbose, preprocess=False)
    return

def create_data_folder(base_path:Path, jpg=False, verbose=False, preprocess=False):
    """
    Creates a Data folder containing all the DICOM slices in jpg/png format,
    the CT scans are located in the base_path/Data/Images folder and the masks of the said
    CT scans are located at base_path/Data/Masks.
    """
    if preprocess:
        save_path = base_path/"Data/Preprocessed"
    else:
        save_path = base_path/"Data/Raw"

    for i in range(1, 21):
        current_patient_folder = f"3Dircadb1/3Dircadb1.{i}"
        convert_folder(dicom_dir=base_path/current_patient_folder/"PATIENT_DICOM/PATIENT_DICOM", output_dir=save_path/"Images", patient_num=i, jpg=jpg, verbose=verbose, preprocess=preprocess)
        convert_folder(dicom_dir=base_path/current_patient_folder/"MASKS_DICOM/MASKS_DICOM/liver", output_dir=save_path/"Masks", patient_num=i, jpg=jpg, verbose=verbose, preprocess=preprocess)

    return 

def convert_folder(dicom_dir, output_dir, patient_num, jpg=False, verbose=False, preprocess=False):
    """
    Takes a folder containing DICOM images and converts them either
    into a jpeg or a png file and saves the converted files in the 
    output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensures the folder exists
    for filename in os.listdir(dicom_dir):
       elems = filename.split("_", 1)
       dicom_to_png_jpeg(dicom_dir=dicom_dir/filename, filename=f"image_{patient_num}_{elems[1]}", save_path=output_dir, jpg=jpg, verbose=verbose, preprocess=preprocess)
    return 


def dicom_to_png_jpeg(dicom_dir, filename, save_path, jpg=False, verbose=False, preprocess=False, min_hu=-100, max_hu=400):
    """
    Converts the filename file in the dicom_dir to either a jpeg 
    or a png file and saves it in the save_path.
    """
    os.makedirs(save_path, exist_ok=True)
    try:
        if preprocess:
            pixel_data = dicom_preprocess(dicom_dir, min_hu=min_hu, max_hu=max_hu)
        else:
            pixel_data = dcmread(dicom_dir).pixel_array.astype(np.float32)
    except Exception as e:
        print(f"{dicom_dir} is not a valid DICOM: {e}")
        return
    
    pixel_data = (pixel_data - min_hu)/(max_hu - min_hu)*255.0
    pixel_data = pixel_data.astype(np.uint8)
    
    image = Image.fromarray(pixel_data)
    ext = 'jpg' if jpg else 'png'
    output_path = os.path.join(save_path, filename+f'.{ext}')
    image.save(output_path)

    if verbose:
        print(f"Converted {dicom_dir} to {output_path}")
        plt.imshow(pixel_data, cmap='gray')
        plt.axis('off')
        plt.show()
    return

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

# Load all the liver present dicom pathes from the dataset -------------------------------------------
def load_all_liver_appearances(base_path:Path, num_patients=[]):
    global LIVER_INDICES
    images, masks = [], []

    to_load = LIVER_INDICES.keys() if num_patients == [] else num_patients
    
    for num_patient in to_load:
        first, last = LIVER_INDICES[num_patient]
        for ind in range(first, last+1):
            images.append(base_path/f"3Dircadb1/3Dircadb1.{num_patient}/PATIENT_DICOM/PATIENT_DICOM/image_{ind}")
            masks.append(base_path/f"3Dircadb1/3Dircadb1.{num_patient}/MASKS_DICOM/MASKS_DICOM/liver/image_{ind}")
            
    return images, masks

