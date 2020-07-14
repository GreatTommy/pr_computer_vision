"""Computes and displays precision and recall graphs.

The human precision and recall are computed on the whole labels dataset (i.e. trainset + devset).
Florent's and Thomas' labels alternatively (and virtually) stand for the ground truth and detection results.
The resulting plot aims at making the decision on whether to focus on diminishing bias or variance.
Its main function is compare_all_precisions_and_recalls().

"""

import os
import glob
import sys
import json
import shutil
from typing import NoReturn

import cv2
import matplotlib.pyplot as plt

from .datasets_preparation import _list_images_without_png

USER_PATH = os.getcwd()
print("USER_PATH : ", USER_PATH)

HUMAN_CALCULATOR_INPUT_DIR = "human_calculator_input"
HUMAN_CALCULATOR_INPUT_DIR_PATH = os.path.join(USER_PATH, HUMAN_CALCULATOR_INPUT_DIR)
HUMAN_CALCULATOR_OUTPUT_FILE = "human_calculator_output.txt"
GT_DIR_PATH = os.path.join(USER_PATH, HUMAN_CALCULATOR_INPUT_DIR, "ground-truth")
DR_DIR_PATH = os.path.join(USER_PATH, HUMAN_CALCULATOR_INPUT_DIR, "detection-results")


def _take_closest(sorted_list, ref):
    """Returns the index of the closest value to ref in sorted_list.

    If two numbers are equally close, returns the index of the smallest.

    """
    min_element = min(sorted_list, key=lambda x: abs(x - ref))
    min_element_index = sorted_list.index(min_element)
    return min_element_index


def _list_labels_without_txt(directory_path):
    """Returns a list of all the labels in directory_path, without the .txt
    extension.
    """
    labels = os.listdir(f"{directory_path}")
    labels_without_txt = []

    for label in labels:
        label = label.strip(".txt")
        labels_without_txt.append(label)

    return labels_without_txt


def _copy_alternatively(
    labels_without_txt, labels_count, labels_src_path, dir_1_path, dir_2_path
):
    """Effectively copies as requested 
    by _copy_labels_alternatively_into_mAP_calculator_input_folder().
    """
    for i in range(labels_count):
        label_source_file = os.path.join(
            labels_src_path, f"{labels_without_txt[i]}.txt"
        )
        if i % 2 == 0:
            shutil.copy(label_source_file, dir_1_path)
        else:
            shutil.copy(label_source_file, dir_2_path)


def _copy_labels_alternatively_into_mAP_calculator_input_folder(
    labels_src_path1, labels_src_path2
):
    """Copies Florent's and Thomas' labels alternatively into the folder
    used to compute the human precision/recall point.
    """
    if os.path.exists(HUMAN_CALCULATOR_INPUT_DIR_PATH):
        shutil.rmtree(HUMAN_CALCULATOR_INPUT_DIR_PATH)
    os.mkdir(HUMAN_CALCULATOR_INPUT_DIR_PATH)
    os.mkdir(f"{HUMAN_CALCULATOR_INPUT_DIR_PATH}{os.sep}ground-truth")
    os.mkdir(f"{HUMAN_CALCULATOR_INPUT_DIR_PATH}{os.sep}detection-results")

    labels_without_txt = _list_labels_without_txt(labels_src_path2)
    labels_count = len(labels_without_txt)

    _copy_alternatively(
        labels_without_txt, labels_count, labels_src_path1, GT_DIR_PATH, DR_DIR_PATH
    )
    _copy_alternatively(
        labels_without_txt, labels_count, labels_src_path2, DR_DIR_PATH, GT_DIR_PATH
    )


def _convert_yolo_coordinates_to_voc(
    x_c_n, y_c_n, width_n, height_n, img_width, img_height
):
    """Returns coordinates converted from YOLO to VOC format."""
    # Remove normalization given the size of the image
    x_c = float(x_c_n) * img_width
    y_c = float(y_c_n) * img_height
    width = float(width_n) * img_width
    height = float(height_n) * img_height

    # Compute half width and half height
    half_width = width / 2
    half_height = height / 2

    # Compute left, top, right, bottom
    # In the official VOC challenge the top-left pixel in the image has coordinates (1;1)
    left = int(x_c - half_width) + 1
    top = int(y_c - half_height) + 1
    right = int(x_c + half_width) + 1
    bottom = int(y_c + half_height) + 1
    return left, top, right, bottom


def _convert_Yolo_labels_to_Voc(directory_path, images_src_path):
    """Converts the labels' format from VOC to YOLO."""
    OBJ_LIST = ["blob", "dot"]

    labels_without_txt = _list_labels_without_txt(directory_path)
    for label in labels_without_txt:
        img_file_path = os.path.join(images_src_path, f"{label}.png")
        img = cv2.imread(img_file_path)
        img_height, img_width = img.shape[:2]

        label_file = f"{directory_path}{os.sep}{label}.txt"

        with open(label_file, "r") as file:
            content = file.readlines()
        content = [x.strip() for x in content]

        with open(label_file, "w") as new_file:
            for line in content:

                # "c" stands for center and "n" stands for normalized
                obj_id, x_c_n, y_c_n, width_n, height_n = line.split()
                obj_name = OBJ_LIST[int(obj_id)]
                left, top, right, bottom = _convert_yolo_coordinates_to_voc(
                    x_c_n, y_c_n, width_n, height_n, img_width, img_height
                )

                # DR_DIR requires a confidence value which is arbitrarily set to 1
                if directory_path == DR_DIR_PATH:
                    new_file.write(
                        f"{obj_name} 1 {str(left)} {str(top)} {str(right)} {str(bottom)}"
                        + "\n"
                    )
                else:
                    new_file.write(
                        f"{obj_name} {str(left)} {str(top)} {str(right)} {str(bottom)}"
                        + "\n"
                    )


def _file_lines_to_list(path):
    """Convert the lines of a file to a list."""
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def _error(msg):
    """Throws an error and exits."""
    print(msg)
    sys.exit(0)


def _compute_human_precision_and_recall():
    """Computes precision and recall and writes it into HUMAN_CALCULATOR_OUTPUT_FILE."""
    MIN_OVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)

    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    # ground-truth
    #     Load each of the ground-truth files into a temporary ".json" file.
    #     Create a list of all the class names present in the ground-truth (gt_classes).

    ground_truth_files_list = glob.glob(GT_DIR_PATH + "/*.txt")
    if len(ground_truth_files_list) == 0:
        _error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()

    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_DIR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            _error(error_msg)
        lines_list = _file_lines_to_list(txt_file)

        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += 'by running the script "remove_space.py" or "rename_class.py" in the "extra/" folder.'
                _error(error_msg)

            # check if class is in the ignore list, if yes skip
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append(
                    {
                        "class_name": class_name,
                        "bbox": bbox,
                        "used": False,
                        "difficult": True,
                    }
                )
                is_difficult = False
            else:
                bounding_boxes.append(
                    {"class_name": class_name, "bbox": bbox, "used": False}
                )
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, "w") as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)

    # detection-results
    #     Load each of the detection-results files into a temporary ".json" file.

    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_DIR_PATH + "/*.txt")
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_DIR_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    _error(error_msg)
            lines = _file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    _error(error_msg)
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append(
                        {"confidence": confidence, "file_id": file_id, "bbox": bbox}
                    )
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x["confidence"]), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)

    # Calculate the AP for each class

    dicoPrecisionRecall = {}  # Format : {classe : (P,R)}

    with open(HUMAN_CALCULATOR_OUTPUT_FILE, "w") as output_file:
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0

            # Load detection-results of that class
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            # Assign detection-results to ground-truth objects
            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]

                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [
                            max(bb[0], bbgt[0]),
                            max(bb[1], bbgt[1]),
                            min(bb[2], bbgt[2]),
                            min(bb[3], bbgt[3]),
                        ]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (
                                (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                                + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                                - iw * ih
                            )
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign detection as true positive/don't care/false positive
                if ovmax >= MIN_OVERLAP:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, "w") as file:
                                file.write(json.dumps(ground_truth_data))

                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1

                else:
                    # false positive
                    fp[idx] = 1

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            rec = tp[:]

            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

            print(f"Human precision for {class_name}s : ", prec[-1])
            print(f"Human recall for {class_name}s : ", rec[-1], "\n")

            dicoPrecisionRecall[class_name] = (prec[-1], rec[-1])

            output_file.write(
                f"{class_name}" + "\n" + f"{prec[-1]}" + "\n" + f"{rec[-1]}" + "\n"
            )

    shutil.rmtree(TEMP_FILES_PATH)
    shutil.rmtree(HUMAN_CALCULATOR_INPUT_DIR_PATH)


def _plot_all_precisions_and_recalls(
    class_name,
    trainset_recall_list,
    trainset_precision_list,
    devset_recall_list,
    devset_precision_list,
    human_recall_dot,
    human_precision_dot,
):
    """Plots the trainset and devset precision-recall curves 
    and the human precision-recall dot on the same graph for comparison."""
    plt.plot(trainset_recall_list, trainset_precision_list, "-b", label="train")
    plt.plot(devset_recall_list, devset_precision_list, "-r", label="dev")
    plt.plot(human_recall_dot, human_precision_dot, "go", label="human")
    plt.axis([0, 1.1, 0, 1.1])
    plt.legend(loc="lower left")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(class_name)
    plt.grid()
    plt.savefig(f"{class_name}.pdf")
    plt.show()
    plt.clf()


def _reformat_string_to_list(string):
    """Transforms a string containing parasitic characters into a well-formed list."""
    for char in "[]":
        string = string.replace(char, "")
    reformatted_list = string.split()
    reformatted_list = list(map(float, reformatted_list))
    return reformatted_list


def _read_file_to_statistics(lines_already_read):
    """Reads the files storing precision and recall values and puts them in lists and variables."""
    with open("trainsetPrecisionRecall.txt", "r") as trainset_file, open(
        "devsetPrecisionRecall.txt", "r"
    ) as devset_file, open(HUMAN_CALCULATOR_OUTPUT_FILE, "r") as human_file:
        for _ in range(lines_already_read):  # Reach the last positions in all files
            trainset_file.readline()
            devset_file.readline()
            human_file.readline()

        trainset_statistic_string = trainset_file.readline()
        trainset_statistic_list = _reformat_string_to_list(trainset_statistic_string)

        devset_statistic_string = devset_file.readline()
        devset_statistic_list = _reformat_string_to_list(devset_statistic_string)

        human_statistic_dot = human_file.readline()
        human_statistic_dot = human_statistic_dot.strip()
        human_statistic_dot = float(human_statistic_dot)

    return trainset_statistic_list, devset_statistic_list, human_statistic_dot


def compare_all_precisions_and_recalls(
    labels_src_dir1: str = "labelsF",
    labels_src_dir2: str = "labelsT",
    images_src_dir: str = "images",
) -> NoReturn:
    """Computes the human precision and recall and plots them 
    with the trainset and devset precisions and recalls outputted by YOLO.

    Parameters
    ----------
    labels_src_dir1, labels_src_dir2 : str, optional
        Name of the directories containing the input labels. 
        The default are "labelsF" and "labelsT".
    images_src_dir : str, optional
        Name of the directory containing the input images.
        The default is "images".

    """
    if not isinstance(labels_src_dir1, str) or not isinstance(labels_src_dir2, str):
        raise TypeError("Labels directories names must be strings.")
    if not isinstance(images_src_dir, str):
        raise TypeError("Images directory name must be a string.")

    if not os.path.exists(
        f"{USER_PATH}{os.sep}{labels_src_dir1}"
    ) or not os.path.exists(f"{USER_PATH}{os.sep}{labels_src_dir2}"):
        raise ValueError("Labels directories not defined.")
    if not os.path.exists(f"{USER_PATH}{os.sep}{images_src_dir}"):
        raise ValueError("Images directory not defined.")
    if not os.path.exists(
        f"{USER_PATH}{os.sep}trainsetPrecisionRecall.txt"
    ) or not os.path.exists(f"{USER_PATH}{os.sep}devsetPrecisionRecall.txt"):
        raise ValueError(
            "trainsetPrecisionRecall.txt or devsetPrecisionRecall.txt files not defined."
        )

    LABELS_DIR_1_PATH = os.path.join(USER_PATH, labels_src_dir1)
    LABELS_DIR_2_PATH = os.path.join(USER_PATH, labels_src_dir2)
    IMG_PATH = os.path.join(USER_PATH, images_src_dir)

    _copy_labels_alternatively_into_mAP_calculator_input_folder(
        LABELS_DIR_1_PATH, LABELS_DIR_2_PATH
    )

    for directory_path in [GT_DIR_PATH, DR_DIR_PATH]:
        _convert_Yolo_labels_to_Voc(directory_path, IMG_PATH)

    _compute_human_precision_and_recall()

    # Read the previously-obtained results and put them into lists
    lines_already_read = 0
    for class_name in ["blob", "dot"]:
        with open("trainsetPrecisionRecall.txt", "r") as trainset_file, open(
            "devsetPrecisionRecall.txt", "r"
        ) as devset_file, open(HUMAN_CALCULATOR_OUTPUT_FILE, "r") as human_file:
            # Lines with class names are ignored
            trainset_file.readline()
            devset_file.readline()
            human_file.readline()
        lines_already_read += 1
        (
            trainset_precision_list,
            devset_precision_list,
            human_precision_dot,
        ) = _read_file_to_statistics(lines_already_read)
        lines_already_read += 1
        (
            trainset_recall_list,
            devset_recall_list,
            human_recall_dot,
        ) = _read_file_to_statistics(lines_already_read)
        lines_already_read += 1

        _plot_all_precisions_and_recalls(
            class_name,
            trainset_recall_list,
            trainset_precision_list,
            devset_recall_list,
            devset_precision_list,
            human_recall_dot,
            human_precision_dot,
        )

    os.remove(f"{USER_PATH}{os.sep}{HUMAN_CALCULATOR_OUTPUT_FILE}")
