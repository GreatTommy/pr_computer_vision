"""Prepares the datasets' directories used to train and evaluate YOLO,
as well as the .data file.
Its main function is create_trainset_and_devset().
"""

import os
import shutil
import random
from typing import NoReturn

from PIL import Image

from .cropping_functions import crop_traces


USER_PATH = os.getcwd()
print("USER_PATH : ", USER_PATH)

COLAB_DATA_PATH = os.path.join(os.sep, "content", "yolov3", "yolo_pr")


def _list_images_without_png(directory_path):
    """Returns a list of all the images in directory_path,
    without the .png extension.
    """
    images = os.listdir(f"{directory_path}")
    images_without_png = []

    for image in images:
        image = image.strip(".png")
        images_without_png.append(image)

    return images_without_png


def alter_images(
    directory_path: str,
    crop: bool = False,
    gray_shades: bool = True,
    resize: bool = False,
) -> NoReturn:
    """Convert the images to grayscale and/or resize them and/or crop them.
    You should crop only if your source images contain 10 drawings. 
    
    Parameters
    ----------
    directory_path : str
        Name of the directory containing the images to alter.
    crop : bool, optional
        If True, crops the images. 
        The default is False.
    gray_shades : bool, optional
        If True, converts the images to grayscale. 
        The default is True.
    resize : bool, optional
        If True, resizes the images. 
        The default is False.

    """
    OUTPUT_WIDTH = 512
    OUTPUT_LENGTH = 512

    if crop:
        images = os.listdir(directory_path)
        print("Cropping the images ...")
        for image in images:
            base = image.split(".")
            altered_image = Image.open(f"{directory_path}{os.sep}{image}")
            if altered_image.size != (1440, 810):
                altered_image = altered_image.resize((1440, 810))
            crops, idx = crop_traces(altered_image)
            for num in idx:
                new_image = crops[num]
                crop_name = f"{base[0]}_{str(num + 1)}.{base[1]}"
                new_image.save(f"{directory_path}{os.sep}{crop_name}")
            altered_image.close()
            os.remove(f"{directory_path}{os.sep}{image}")
    if gray_shades or resize:
        images = os.listdir(directory_path)
        print("Altering the images ...")
        for image in images:
            altered_image = Image.open(f"{directory_path}{os.sep}{image}")
            if gray_shades:
                altered_image = altered_image.convert("L")
            if resize:
                altered_image = altered_image.resize((OUTPUT_WIDTH, OUTPUT_LENGTH))
            altered_image.save(f"{directory_path}{os.sep}{image}")


def _copy_into_trainset(
    labels_src_path1, labels_src_path2, images_src_path, train_dest_path, images_names
):
    """Copies images and labels into train_dest_path."""
    print(f"Copying images and labels to {train_dest_path} ...")
    images_initial_amount = len(images_names)

    images_dest_dir = os.path.join(train_dest_path, "images")

    train_dest_dir = os.path.dirname(train_dest_path)
    trainset_inventory_file = os.path.join(train_dest_path, f"{train_dest_dir}.txt")
    with open(trainset_inventory_file, "w") as trainset_inventory_file:
        while len(images_names) > 0.2 * images_initial_amount:
            image_name = images_names[0]

            image_src_file = os.path.join(images_src_path, f"{image_name}.png")
            shutil.copy(image_src_file, images_dest_dir)

            # Double-labelling requires double images with '-bis' marker
            image_to_copy = os.path.join(images_dest_dir, f"{image_name}.png")
            copied_image = os.path.join(images_dest_dir, f"{image_name}-bis.png")
            shutil.copyfile(image_to_copy, copied_image)

            labels_src_file1 = os.path.join(labels_src_path1, f"{image_name}.txt")
            labels_src_file2 = os.path.join(labels_src_path2, f"{image_name}.txt")
            labels_dest_file1 = os.path.join(
                train_dest_path, "labels", f"{image_name}.txt"
            )
            labels_dest_file2 = os.path.join(
                train_dest_path, "labels", f"{image_name}-bis.txt"
            )

            shutil.copyfile(labels_src_file1, labels_dest_file1)
            shutil.copyfile(labels_src_file2, labels_dest_file2)

            # Inventory files must contain the names of the sets' images
            image_to_copy_Colab_path = os.path.join(
                COLAB_DATA_PATH,
                f"{train_dest_dir}",
                "images",
                f"{images_names[0]}.png",
            )
            copied_image_Colab_path = os.path.join(
                COLAB_DATA_PATH,
                f"{train_dest_dir}",
                "images",
                f"{images_names[0]}-bis.png",
            )
            trainset_inventory_file.write(f"{image_to_copy_Colab_path}" + "\n")
            trainset_inventory_file.write(f"{copied_image_Colab_path}" + "\n")

            images_names.pop(0)

    alter_images(images_dest_dir)


def _copy_into_devset(
    labels_src_path1, labels_src_path2, images_src_path, dev_dest_path, images_names
):
    """Copies images and labels into dev_dest_path."""
    print(f"Copying images and labels to {dev_dest_path} ...")
    images_left_amount = len(images_names)

    dev_dest_dir = os.path.dirname(dev_dest_path)
    devset_inventory_file = os.path.join(dev_dest_path, f"{dev_dest_dir}.txt")
    with open(devset_inventory_file, "w") as devset_inventory_file:
        for i in range(images_left_amount):
            image_name = images_names[i]

            image_src_file = os.path.join(images_src_path, f"{image_name}.png")
            image_dest_file = os.path.join(dev_dest_path, "images", f"{image_name}.png")
            shutil.copyfile(image_src_file, image_dest_file)

            # Inventory files must contain the names of the sets' images
            image_Colab_path = os.path.join(
                COLAB_DATA_PATH, f"{dev_dest_dir}", "images", f"{image_name}.png"
            )
            devset_inventory_file.write(f"{image_Colab_path}" + "\n")

            # Alternate between Florent's and Thomas' labels
            label_dest_dir = os.path.join(dev_dest_path, "labels")
            if i % 2 == 0:
                label_src_file = os.path.join(labels_src_path1, f"{image_name}.txt")
                shutil.copy(label_src_file, label_dest_dir)
            else:
                label_src_file = os.path.join(labels_src_path2, f"{image_name}.txt")
                shutil.copy(label_src_file, label_dest_dir)

    devset_images_dir = os.path.join(dev_dest_path, "images")
    alter_images(devset_images_dir)


def _create_data_file(train_dest_dir, dev_dest_dir):
    """Creates the .data file with the number of classes, 
    trainset, devset and names of the classes.
    """
    with open("yolo_pr.data", "w") as file:
        file.write("classes=2\n")
        file.write(f"train=yolo_pr/{train_dest_dir}/{train_dest_dir}.txt" + "\n")
        file.write(f"valid=yolo_pr/{dev_dest_dir}/{dev_dest_dir}.txt" + "\n")
        file.write("names=yolo_pr/yolo_pr.names")


def create_trainset_and_devset(
    labels_src_dir1: str = "labelsF",
    labels_src_dir2: str = "labelsT",
    images_src_dir: str = "images",
    train_dest_dir: str = "trainset",
    dev_dest_dir: str = "devset",
) -> NoReturn:
    """Creates the trainset and devset from two labels and one images folders.
    
    Shuffles the whole dataset to avoid bias.
    Copies 80% of the data into the trainset then the remaining 20% into the devset.
    (The trainset is double-labelled while the devset isn't.)
    Finally, the .data file is created accordingly.
    
    Parameters
    ----------
    labels_src_dir1, labels_src_dir2 : str, optional
        Name of the directories containing the input labels. 
        The default are "labelsF" and "labelsT".
    images_src_dir : str, optional
        Name of the directory containing the input images.
        The default is "images".
    train_dest_dir, dev_dest_dir : str, optional
        Name of the trainset and devset directories to create. 
        The default are "trainset" and "devset".
    """
    if not isinstance(labels_src_dir1, str) or not isinstance(labels_src_dir2, str):
        raise TypeError("Labels directories names must be strings.")
    if not isinstance(images_src_dir, str):
        raise TypeError("Images directory name must be a string.")
    if not isinstance(train_dest_dir, str) or not isinstance(dev_dest_dir, str):
        raise TypeError("Trainset and devset directories names must be strings.")

    if not os.exists(f"{USER_PATH}{os.sep}{labels_src_dir1}") or not os.exists(
        f"{USER_PATH}{os.sep}{labels_src_dir2}"
    ):
        raise ValueError("Labels directories not defined.")
    if not os.exists(f"{USER_PATH}{os.sep}{images_src_dir}"):
        raise ValueError("Images directory not defined.")

    LABELS_DIR_1_PATH = os.path.join(USER_PATH, labels_src_dir1)
    LABELS_DIR_2_PATH = os.path.join(USER_PATH, labels_src_dir2)
    IMG_PATH = os.path.join(USER_PATH, images_src_dir)
    TRAINSET_PATH = os.path.join(USER_PATH, train_dest_dir)
    DEVSET_PATH = os.path.join(USER_PATH, dev_dest_dir)

    # The distribution between the devset and trainset must remain the same for devs,
    # so that YOLO's performance isn't wrongly impacted.
    # Therefore a distribution list is created only if none exists.
    names_file = os.path.join(USER_PATH, "names_file.txt")
    if not os.path.exists(names_file):
        images_names = _list_images_without_png(IMG_PATH)
        random.shuffle(images_names)  # to diversify the sets
        with open("names_file.txt", "w") as file:
            print("Writing into names_file.txt ...")
            for image_name in images_names:
                file.write(f"{image_name}" + "\n")
    else:
        with open("names_file.txt", "r") as file:
            print("Reading from names_file.txt ...")
            images_names = file.readlines()
            images_names = [image_name.strip() for image_name in images_names]

    if os.path.exists(TRAINSET_PATH):
        shutil.rmtree(TRAINSET_PATH)
    os.mkdir(TRAINSET_PATH)
    os.mkdir(f"{TRAINSET_PATH}{os.sep}images")
    os.mkdir(f"{TRAINSET_PATH}{os.sep}labels")
    if os.path.exists(DEVSET_PATH):
        shutil.rmtree(DEVSET_PATH)
    os.mkdir(DEVSET_PATH)
    os.mkdir(f"{DEVSET_PATH}{os.sep}images")
    os.mkdir(f"{DEVSET_PATH}{os.sep}labels")

    _copy_into_trainset(
        LABELS_DIR_1_PATH, LABELS_DIR_2_PATH, IMG_PATH, TRAINSET_PATH, images_names
    )
    _copy_into_devset(
        LABELS_DIR_1_PATH, LABELS_DIR_2_PATH, IMG_PATH, DEVSET_PATH, images_names
    )

    _create_data_file(train_dest_dir, dev_dest_dir)

    print("Your trainset, devset and .data file are now ready.")
