"""Utilitary functions for the Colab notebook."""

import os
from typing import NoReturn

from google.colab import files


def remove_labelled_images(directory_to_clean: str) -> NoReturn:
    """Removes the .png files from directory_to_clean
    since only .txt files are later used by the rating algorithm.

    Parameters
    ----------
    directory_to_clean : str
        The directory containing the results outputted by YOLO.

    """
    files_to_remove = os.listdir(directory_to_clean)
    for file in files_to_remove:
        if file.split(".")[1] == "png":
            os.remove(f"{directory_to_clean}/{file}")


def _list_labelled_images_to_remove(directory_to_clean):
    """Lists the .png files from directory_to_clean."""
    images_to_remove = ""
    images = os.listdir(directory_to_clean)
    for image in images:
        if image.split(".")[1] == "png":
            images_to_remove += f"{directory_to_clean}{os.sep}{image} "

    return images_to_remove
