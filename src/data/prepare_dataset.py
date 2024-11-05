from src.align import align_single_face
from glob import glob
import cv2
import numpy as np
import os
from tqdm import tqdm

def prepare_dataset(dataset_path: str) -> None:
    """
    Prepares a dataset by aligning all face images.
    """
    image_save_path = f"{dataset_path}/aligned"
    os.makedirs(image_save_path, exist_ok=True)
    image_paths = glob(f"{dataset_path}/*.bmp")
    for image_path in tqdm(image_paths, desc="Aligning faces"):
        print(image_path)
        image = cv2.imread(image_path)
        aligned_image = align_single_face(image)
        if aligned_image is not None:
            cv2.imwrite(f"{image_save_path}/{os.path.basename(image_path)}", aligned_image)

if __name__ == "__main__":
    prepare_dataset("/Users/harsh/Documents/github/caire/src/data/CodingImages")
