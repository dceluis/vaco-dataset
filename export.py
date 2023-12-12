import requests
import sys
import json
import os
import random
import zipfile
import glob
import shutil
from sahi.slicing import slice_coco
from tqdm import tqdm
import wandb

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

        dataset_dir = f"./export/{version}/{dataset}"
        coco_annotation_file_path = f"{dataset_dir}/result.json"

        # unzip the only the result.json file from the zip file
        with zipfile.ZipFile(f"{title}.zip", 'r') as zip_ref:
            zip_ref.extract("result.json", dataset_dir)

        # remove the zip file
        os.remove(f"{title}.zip")

        coco_dict = json.load(open(coco_annotation_file_path))

        return coco_dict, coco_annotation_file_path
    else:
        print("Failed to start the export.")

def slice_dataset(dataset_dir):
    coco_annotation_file_path = f"{dataset_dir}/result.json"

    # slice the dataset
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

    return coco_dict, coco_annotation_file_path

def convert_dataset(dataset_dir):
    coco_annotation_file_path = f"{dataset_dir}/result.json"

    # first, increase the category ids by 1
    # JSON2YOLO subtracts 1 from the category ids, for some reason
    coco_dict = json.load(open(coco_annotation_file_path))
    for category in coco_dict["categories"]:
        category["id"] += 1
    for annotation in coco_dict["annotations"]:
        annotation["category_id"] += 1

    # save the new json file
    with open(coco_annotation_file_path, "w") as f:
        json.dump(coco_dict, f)

    convert_coco_json(
        json_dir=dataset_dir,
    )

    # move the converted files to the correct directory
    files_pattern = f"./new_dir/labels/result/*.txt"

    labels_dir = os.path.join(dataset_dir, "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for file in glob.glob(files_pattern):
        shutil.move(file, os.path.join(labels_dir, os.path.basename(file)))

    # remove the new_dir folder
    shutil.rmtree("./new_dir")

    return coco_dict, coco_annotation_file_path

def log_artifact(dataset_dir, dataset_name):
    wandb.login()
    run = wandb.init(project="YOLOv8", job_type="dataset")

    artifact = wandb.Artifact(
        name = dataset_name,
        type = "dataset"
    )

    artifact.add_dir(dataset_dir)
    wandb.log_artifact(artifact)
    run.finish()

def run(version = 1):
    # create a new folder for the export
    if not os.path.exists(f"./export/{version}"):
        os.mkdir(f"./export/{version}")

        # download the train, val and test datasets
        train_dict, train_json_path = download_dataset("train", version)
        val_dict, val_json_path = download_dataset("val", version)
        test_dict, test_json_path = download_dataset("test", version)

        # slice the datasets
        slice_dataset(os.path.dirname(train_json_path))
        slice_dataset(os.path.dirname(val_json_path))
        slice_dataset(os.path.dirname(test_json_path))

        # convert the datasets to yolo format
        convert_dataset(os.path.dirname(train_json_path))
        convert_dataset(os.path.dirname(val_json_path))
        convert_dataset(os.path.dirname(test_json_path))

        # merge the classes from the train, val and test datasets
        classes = []
        for dataset in [train_dict, val_dict, test_dict]:
            for category in dataset["categories"]:
                classes.append(category["name"])

        classes = list(set(classes))

        # create the data.yaml file
        with open(f"./export/{version}/data.yaml", "w") as f:
            f.write(f"train: ./train/images\n")
            f.write(f"val: ./val/images\n")
            f.write(f"test: ./test/images\n")
            f.write(f"\n")
            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {classes}\n")

        no_annotations_dir = f"./export/{version}/no_annotations"
        # make a folder for images with no annotations
        if not os.path.exists(no_annotations_dir):
            os.mkdir(no_annotations_dir)

        # move all images with no annotations
        for image in tqdm(glob.glob(f"./export/{version}/**/images/*")):
            ext = os.path.splitext(image)[-1]
            if not os.path.exists(image.replace("images", "labels").replace(ext, ".txt")):
                # shutil.move(image, os.path.join(no_annotations_dir, os.path.basename(image)))
                os.remove(image)

    # log the dataset as an artifact
    dataset_name = "vacocam_dataset"
    log_artifact(f"./export/{version}", dataset_name)

if __name__ == "__main__":
    # create the export folder if it doesn't exist, otherwise leave it as is
    if not os.path.exists("./export"):
        os.mkdir("./export")

    version = 1
    if len(sys.argv) == 1:
        # generate an ascending version number, always one more than the last one in the ./export folder
        for folder in glob.glob("./export/*"):
            if int(folder.split("/")[-1]) >= version:
                version = int(folder.split("/")[-1]) + 1
    else:
        version = int(sys.argv[1])

    run(version)
