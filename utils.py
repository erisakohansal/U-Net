import os
import numpy as np
import zipfile
from pydicom import dcmread
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path #https://docs.python.org/3/library/pathlib.html
import shutil

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

def train_test_division(base_path:Path, jpg=False, verbose=False, train_percent=80, total_images=2823):
    train_count = np.floor(total_images * train_percent/100)
    test_count = total_images - train_count
    print("total:", total_images)
    print("train:", train_count)
    print("test:", test_count)

    os.makedirs(base_path/"Data/Train/Images", exist_ok=True)  
    os.makedirs(base_path/"Data/Test/Images", exist_ok=True) 
    os.makedirs(base_path/"Data/Train/Masks", exist_ok=True)  
    os.makedirs(base_path/"Data/Test/Masks", exist_ok=True) 

    counter = 0
    current_folder = "Train"
    for num_patient in range(1, 21):
        for filename in os.listdir(base_path/f"3Dircadb1/3Dircadb1.{num_patient}/PATIENT_DICOM/PATIENT_DICOM"):
            elems = filename.split("_", 1)
            num_slice = elems[1]
            counter += 1
            if counter == train_count + 1: #current_folder == "train" and counter > train_count:  
                current_folder = "Test"
            dicom_to_png_jpeg(dicom_dir=base_path/f"3Dircadb1/3Dircadb1.{num_patient}/PATIENT_DICOM/PATIENT_DICOM/image_{num_slice}", filename=f"image_{num_patient}_{num_slice}", save_path=base_path/"Data"/current_folder/"Images", jpg=jpg, verbose=verbose, preprocess=True)
            dicom_to_png_jpeg(dicom_dir=base_path/f"3Dircadb1/3Dircadb1.{num_patient}/MASKS_DICOM/MASKS_DICOM/liver/image_{num_slice}", filename=f"image_{num_patient}_{num_slice}", save_path=base_path/"Data"/current_folder/"Masks", jpg=jpg, verbose=verbose, preprocess=False)
        
    assert len(os.listdir(base_path/"Data/Train/Images")) == train_count and len(os.listdir(base_path/"Data/Train/Masks")) == train_count
    assert len(os.listdir(base_path/"Data/Test/Images")) == test_count and len(os.listdir(base_path/"Data/Test/Images")) == test_count

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


def dicom_to_png_jpeg(dicom_dir, filename, save_path, jpg=False, verbose=False, preprocess=False):
    """
    Converts the filename file in the dicom_dir to either a jpeg 
    or a png file and saves it in the save_path.
    """
    os.makedirs(save_path, exist_ok=True)
    try:
        if preprocess:
            pixel_data = dicom_preprocess(dicom_dir)
        else:
            pixel_data = dcmread(dicom_dir).pixel_array.astype(np.float32)
    except Exception as e:
        print(f"{dicom_dir} is not a valid DICOM: {e}")
        return
    
    min_val, max_val = pixel_data.min(), pixel_data.max()
    # Avoid division by zero
    if max_val > min_val:
        pixel_data = (pixel_data - min_val) / (max_val - min_val) * 255.0
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

# Loading one folder -------------------------------------------------------------------------------
def load_folder(folder_path: Path):
    files = []

    for img in folder_path.iterdir():
        #print(f"iterdir:{img}")
        if img.is_file():
            files.append(str(img))

    return files

# Loading all the patients -------------------------------------------------------------------------
def load_3Dircadb(base_path: str):
    """
    """
    patient_extension_path = Path("PATIENT_DICOM") / "PATIENT_DICOM"
    masks_extension_path = Path("MASKS_DICOM") / "MASKS_DICOM" / "liver"
    #labelled_extension_path = Path("LABELLED_DICOM") / "LABELLED_DICOM"

    all_patient = []
    all_masks = []
    #all_labelled = []

    for current_path in base_path.iterdir(): # patients are not in order, problem???????? sort????
        print(f"current:{current_path}")
        if current_path.is_dir():
            patient_path = current_path / patient_extension_path
            masks_path = current_path / masks_extension_path
            #labelled_path = current_path / labelled_extension_path

            # the rest is the same as load_patient_dicom, we don't use the load_dicom file because
            # appending is faster and more efficient than concatenating
            for img in patient_path.iterdir(): # same number of elements in each of the three folders, one loop is ok
                print(f"img:{img}")
                if img.is_file():
                    all_patient.append(str(img))
                    image_name = os.path.splitext(img)[0]
                    all_masks.append(str(masks_path / image_name))
                    #all_labelled.append(str(labelled_path / image_name))
                    

    return all_patient, all_masks #, all_labelled
