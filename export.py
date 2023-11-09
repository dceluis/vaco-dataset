import requests
import sys
import json
import os
import random
import zipfile
import glob
import shutil
from sahi.slicing import slice_coco

sys.path.append('JSON2YOLO')
from general_json2yolo import convert_coco_json

# Set up the necessary variables
label_studio_host = os.environ.get("LABEL_STUDIO_HOST", "http://localhost:8080")
label_studio_access_token = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN", "your_access_token")

dataset_dict = {
    "train": 57,
    "val": 58,
    "test": 59,
}

def download_dataset(dataset, version: int = 1):
    # Make a POST request to start the export
    headers = {
        "Authorization": f"Token {label_studio_access_token}",
        "Content-Type": "application/json",
    }

    # Make a POST request to start the export
    response = requests.post(
        f"{label_studio_host}/api/projects/3/exports/",
        headers=headers,
        data=json.dumps({"task_filter_options": {"view": dataset_dict[dataset]}}),
    )

    # Check if the request was successful
    if response.status_code == 201:
        response_data = response.json()
        export_id = response_data['id']
        title = f"{response_data['title']}_{dataset}"

        download_response = requests.get(
            f"{label_studio_host}/api/projects/3/exports/{export_id}/download?exportType=COCO",
            headers=headers,
            stream=True
        )

        # Check if the download request was successful
        if download_response.status_code == 200:
            # Write the content of the download to a file
            with open(f"{title}.zip", 'wb') as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Export {title}.zip downloaded successfully.")
        else:
            print("Failed to download the coco export.")
            exit(1)

        # unzip the only the result.json file from the zip to the ./export version folder
        with zipfile.ZipFile(f"{title}.zip", 'r') as zip_ref:
            zip_ref.extract("result.json", f"./export/{version}/{dataset}")

        # remove the zip file
        os.remove(f"{title}.zip")

        dataset_dir = f"./export/{version}/{dataset}"
        coco_annotation_file_path = f"{dataset_dir}/result.json"

        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir="/",
            output_coco_annotation_file_name=f"result_sliced",
            ignore_negative_samples=False,
            output_dir=f"{dataset_dir}/images",
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            min_area_ratio=0.1,
            verbose=False
        )

        # remove the result.json file
        os.remove(coco_annotation_file_path)

        # move coco file
        shutil.move(coco_path, coco_annotation_file_path)

        new_coco_path = os.path.join(dataset_dir, os.path.basename(coco_path))

        return coco_dict, new_coco_path
    else:
        print("Failed to start the export.")

def convert_dataset(dataset_dir):
    convert_coco_json(
        json_dir=dataset_dir,
    )

    # move the converted files to the correct directory
    files_pattern = f"./new_dir/labels/result/*.txt"

    labels_dir = os.path.join(dataset_dir, "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for file in glob.glob(files_pattern):
        shutil.move(file, labels_dir)

# create the export folder if it doesn't exist, otherwise leave it as is
if not os.path.exists("./export"):
    os.mkdir("./export")

# generate an ascending version number, always one more than the last one in the ./export folder
version = 1
for folder in glob.glob("./export/*"):
    if int(folder.split("/")[-1]) >= version:
        version = int(folder.split("/")[-1]) + 1

# create a new folder for the export
os.mkdir(f"./export/{version}")

# download the train, val and test datasets
train_dict, train_json_path = download_dataset("train", version)

val_dict, val_json_path = download_dataset("val", version)

test_dict, test_json_path = download_dataset("test", version)

convert_dataset(f"./export/{version}/train")
convert_dataset(f"./export/{version}/val")
convert_dataset(f"./export/{version}/test")
