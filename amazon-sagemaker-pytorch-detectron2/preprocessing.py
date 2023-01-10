# preprocessing.py
import argparse
import os
import json
import re
import numpy as np
import tempfile

import pandas as pd
from sklearn.model_selection import train_test_split
import random
from datetime import datetime
import boto3


os.system("pip install pathlib")
os.system("pip install typing")

from pathlib import Path
from typing import Mapping, Optional, Sequence
data_dir = "/opt/ml/processing/input"

def upload_annotations(p_annotations, p_channel: str,bucket,prefix_data):
    rsc_bucket = boto3.resource("s3").Bucket(bucket)

    json_lines = [json.dumps(elem) for elem in p_annotations]
    to_write = "\n".join(json_lines)

    with tempfile.NamedTemporaryFile(mode="w") as fid:
        fid.write(to_write)
        rsc_bucket.upload_file(
            fid.name, f"{prefix_data}/annotations/{p_channel}.manifest"
        )
        
def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r
def create_annotation_channel(
    channel_id: str,
    path_to_annotation: Path,
    bucket_name: str,
    data_prefix: str,
    img_annotation_to_ignore: Optional[Sequence[str]] = None,
) -> Sequence[Mapping]:
    r"""Change format from original to augmented manifest files
    Parameters
    ----------
    channel_id : str
        name of the channel, i.e. training, validation or test
    path_to_annotation : Path
        path to annotation file
    bucket_name : str
        bucket where the data are uploaded
    data_prefix : str
        bucket prefix
    img_annotation_to_ignore : Optional[Sequence[str]]
        annotation from these images are ignore because the corresponding images are corrupted, default to None
    Returns
    -------
    Sequence[Mapping]
        List of json lines, each lines contains the annotations for a single. This recreates the
        format of augmented manifest files that are generated by Amazon SageMaker GroundTruth
        labeling jobs
    """
    if channel_id not in ("training", "validation", "testing"):
        raise ValueError(
            f"Channel identifier must be training, validation or testing. The passed values is {channel_id}"
        )
    if not path_to_annotation.exists():
        raise FileNotFoundError(f"Annotation file {path_to_annotation} not found")

    df_annotation = pd.read_csv(
        path_to_annotation,
    )
#     print("printing df_annotation -")
#     print(df_annotation.columns)
#     print(df_annotation.head())

    df = pd.DataFrame(np.stack(df_annotation['bbox'].apply(lambda x: expand_bbox(x))), columns = ['left', 'top', 'width', 'height'])
    df_annotation = pd.concat([df_annotation, df] ,axis = 1)
    df_annotation.drop(columns=['bbox', 'source'], inplace=True)
    df_annotation['left'] = df_annotation['left'].astype(np.float)
    df_annotation['top'] = df_annotation['top'].astype(np.float)
    df_annotation['width'] = df_annotation['width'].astype(np.float)
    df_annotation['height'] = df_annotation['height'].astype(np.float)
#     print(df_annotation.columns)
#     print(df_annotation.head())

    jsonlines = []
    for img_id in df_annotation["image_id"].unique():
        img_annotations = df_annotation.loc[df_annotation["image_id"] == img_id, :]
        annotations = []
        for (
            _,
            img_width,
            img_heigh,
            bbox_l,
            bbox_t,
            bbox_w,
            bbox_h,
        ) in img_annotations.itertuples(index=False):
            annotations.append(
                {
                    "class_id": 0,
                    "width": bbox_w,
                    "top": bbox_t,
                    "left": bbox_l,
                    "height": bbox_h,
                }
            )
        jsonline = {
            "wheat": {
                "annotations": annotations,
                "image_size": [{"width": img_width, "depth": 3, "height": img_heigh,}],
            },
            "wheat-metadata": {
                "job_name": f"labeling-job/wheat-{channel_id}",
                "class-map": {"0": "wheat"},
                "human-annotated": "yes",
                "objects": len(annotations) * [{"confidence": 0.0}],
                "type": "groundtruth/object-detection",
                "creation-date": datetime.now()
                .replace(second=0, microsecond=0)
                .isoformat(),
            },
            "source-ref": f"s3://{bucket_name}/{data_prefix}/{channel_id}/{img_id}",
        }
        jsonlines.append(jsonline)
    return jsonlines


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--prefix", type=str)
    args, _ = parser.parse_known_args()
    bucket = args.bucket
    prefix_data = args.prefix
    
    train_dir = "train"
    print(os.listdir(data_dir))
    file_list = os.listdir(os.path.join(data_dir,train_dir))
    random.shuffle(file_list)
    train_file_list = file_list[:int(0.7 * len(file_list))]
    val_file_list = file_list[int(0.7 * len(file_list)):int(0.85 * len(file_list))]
    test_file_list = file_list[int(0.85 * len(file_list)):]
    print(len(train_file_list) , len(val_file_list), len(test_file_list))

    channel_names = ["training" , "validation","testing"]

    local_folder = "/opt/ml/processing"
    for channel in channel_names:
        if not os.path.exists(os.path.join(local_folder,channel)):
            os.makedirs(os.path.join(local_folder,channel))
            
    for file in train_file_list:
        os.system(f"cp {os.path.join(data_dir,train_dir,file)} {os.path.join(local_folder,channel_names[0],file)}")
    for file in val_file_list:
        os.system(f"cp {os.path.join(data_dir,train_dir,file)} {os.path.join(local_folder,channel_names[1],file)}")
    for file in test_file_list:
        os.system(f"cp {os.path.join(data_dir,train_dir,file)} {os.path.join(local_folder,channel_names[2],file)}")

    for i,channel in enumerate(channel_names):
        print(f"Channel - {channel_names[i]} number of files - {len(os.listdir(os.path.join(local_folder,channel_names[i])))}")

        
    base_annotation_folder = os.path.join(data_dir , "train.csv")
    df_annotations = pd.read_csv(base_annotation_folder)
    df_annotations["image_id"] = df_annotations["image_id"] + ".jpg"
    df_annotations.rename(columns = {"width":"image_width" , "height" : "image_height"} , inplace = True)
    # print(df_annotations.head())
    train_annotations = df_annotations[df_annotations["image_id"].isin(train_file_list)]
    valid_annotations = df_annotations[df_annotations["image_id"].isin(val_file_list)]
    test_annotations = df_annotations[df_annotations["image_id"].isin(test_file_list)]

    train_annotations.to_csv(os.path.join(local_folder , "annotations_train.csv") , index = False)
    valid_annotations.to_csv(os.path.join(local_folder , "annotations_valid.csv") , index = False)
    test_annotations.to_csv(os.path.join(local_folder , "annotations_test.csv") , index = False)
    
    
    annotation_folder = Path(local_folder) 
    channel_to_annotation_path = {
        "training": annotation_folder / "annotations_train.csv",
        "validation": annotation_folder / "annotations_valid.csv",
        "testing" : annotation_folder / "annotations_test.csv",
    }
    channel_to_annotation = {}

    
    for channel in channel_to_annotation_path.keys():
        annotations = create_annotation_channel(
            channel,
            channel_to_annotation_path[channel],
            bucket,
            prefix_data,

        )
        print(f"Number of {channel} annotations: {len(annotations)}")
        channel_to_annotation[channel] = annotations
        
    print(f"Saving annotations to - s3://{bucket}/{prefix_data}/annotations")
    
    for channel_id, annotations in channel_to_annotation.items():
        upload_annotations(annotations, channel_id,bucket,prefix_data)
    