import os
import numpy as np
import zipfile
import pydicom
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path

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


def convert_folder(dicom_dir, output_dir, jpg=False, verbose=False):
    """
    Takes a folder containing DICOM images and converts them either
    into a jpeg or a png file and saves the converted files in the 
    output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensures the folder exists
    for filename in os.listdir(dicom_dir):
       dicom_to_png_jpeg(dicom_dir, filename, output_dir, jpg, verbose)
    return


def dicom_to_png_jpeg(dicom_dir, filename, save_path, jpg=False, verbose=False):
    """
    Converts the filename file in the dicom_dir to either a jpeg 
    or a png file and saves it in the save_path.
    To show a DICOM image we need to preprocess the data sometimes.

    DICOM help:
    https://stackoverflow.com/questions/70091655/dicom-data-training-failed-by-pytorch
    https://www.youtube.com/watch?v=N-3-AOU54yE
    https://mlerma54.github.io/papers/lidc-dicom.pdf

    This task consists of changing the contrast(opacity?) of the image:
    hounsfield_min = pixel_data.min()
    hounsfield_max = pixel_data.max()
    hounsfield_range = hounsfield_max - hounsfield_min
    """

    try:
        pixel_data = dicom_preprocess(dicom_dir / filename)
    except Exception as e:
        print(f"{filename} is not a valid DICOM: {e}")
        return
    
    image = Image.fromarray(pixel_data)
    ext = 'jpg' if jpg else 'png'
    output_path = os.path.join(save_path, os.path.splitext(filename)[0] + f'.{ext}')
    image.save(output_path)

    if verbose:
        print(f"Converted {filename} to {output_path}")
        plt.imshow(pixel_data, cmap='gray')
        plt.axis('off')
        plt.show()
    return

def dicom_preprocess(path):
    dcm_file = pydicom.dcmread(path)
    
    pixel_data = dcm_file.pixel_array.astype(np.float32) # why np.float32?????

    slope = getattr(dcm_file, "RescaleSlope", 1.0)
    intercept = getattr(dcm_file, "RescaleIntercept", 0.0)
    pixel_data = pixel_data * slope + intercept

    
    min_val, max_val = pixel_data.min(), pixel_data.max()
    # Avoid division by zero
    if max_val > min_val: # normalization in [0,1], what about mean/deviation normalization?
        pixel_data = (pixel_data - min_val) / (max_val - min_val) * 255.0
    else:
        pixel_data[:] = 0

    pixel_data = pixel_data.astype(np.uint8)
    return pixel_data

def delete_converted_folders(): pass

"""
if __name__ == "__main__":
    base_path = Path("C:/Users/HP/Desktop/PIMA/3Dircadb1")
    for i in range(2, 3):
        current = f"3Dircadb1.{i}"
        patient = "PATIENT_DICOM"
        masks = "MASKS_DICOM"
        labelled = "LABELLED_DICOM"

        unzip(base_path / current / (masks + ".zip"), base_path / current / masks)
        convert_folder(base_path / current / masks / masks / "liver", base_path / current / masks / "converted")

        unzip(base_path / current / (patient + ".zip"), base_path / current / patient)
        convert_folder(base_path / current / patient / patient, base_path / current / patient / "converted")

        unzip(base_path / current / (labelled + ".zip"), base_path / current / labelled)
        convert_folder(base_path / current / labelled / labelled, base_path / current / labelled / "converted")
"""