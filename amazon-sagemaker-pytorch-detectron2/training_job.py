#!/usr/bin/env python
import requests
import os
import pandas as pd

from sagemaker.analytics import TrainingJobAnalytics
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.sklearn.processing import SKLearnProcessor

import boto3
import botocore
import s3fs
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import ProcessingInput, ProcessingOutput
import json


def main():

    session = sagemaker.Session(boto3.session.Session())

    BUCKET_NAME = os.environ["BUCKET_NAME"]
    PREFIX = os.environ["PREFIX"]
    REGION = os.environ["AWS_DEFAULT_REGION"]
    # Replace with your IAM role arn that has enough access (e.g. SageMakerFullAccess)
    IAM_ROLE_NAME = os.environ["IAM_ROLE_NAME"]
    GITHUB_SHA = os.environ["GITHUB_SHA"]
    ACCOUNT_ID = session.boto_session.client("sts").get_caller_identity()["Account"]
    # Replace with your desired training instance
    training_instance = "ml.g4dn.12xlarge"

    # Replace with your data s3 path

    ## Upload raw data to s3

    s3 = boto3.resource("s3")

    key_prefix = f"detectron2/raw_data/train.csv"
    s3.Bucket(BUCKET_NAME).upload_file("../inputs/train.csv", key_prefix)

    train_dir = "../inputs/train"
    sagemaker_session = sagemaker.Session()
    input_data = sagemaker_session.upload_data(
        train_dir,
        bucket=BUCKET_NAME,
        key_prefix="{}/{}/{}".format("detectron2", "raw_data", "train"),
    )

    input_data = "s3://{}/{}/{}".format(BUCKET_NAME, "detectron2", "raw_data")

    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=IAM_ROLE_NAME,
        instance_type="ml.m5.xlarge",
        instance_count=1,
    )

    sklearn_processor.run(
        code="../src/preprocessing.py",
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/training",
                destination="s3://object-detection-ravi123/detectron2/processed_data/training",
            ),
            ProcessingOutput(
                output_name="validation_data",
                source="/opt/ml/processing/validation",
                destination="s3://object-detection-ravi123/detectron2/processed_data/validation",
            ),
            ProcessingOutput(
                output_name="testing_data",
                source="/opt/ml/processing/testing",
                destination="s3://object-detection-ravi123/detectron2/processed_data/testing",
            ),
        ],
        arguments=[
            "--bucket",
            "object-detection-ravi123",
            "--prefix",
            "detectron2/processed_data",
        ],
    )

    preprocessing_job_description = sklearn_processor.jobs[-1].describe()

    output_config = preprocessing_job_description["ProcessingOutputConfig"]
    for output in output_config["Outputs"]:
        if output["OutputName"] == "train_data":
            preprocessed_training_data = output["S3Output"]["S3Uri"]
        if output["OutputName"] == "test_data":
            preprocessed_test_data = output["S3Output"]["S3Uri"]

    # Data configuration

    prefix_data = f"{PREFIX}/processed_data"
    prefix_report = PREFIX

    training_channel = f"s3://{BUCKET_NAME}/{prefix_data}/training/"
    validation_channel = f"s3://{BUCKET_NAME}/{prefix_data}/validation/"
    test_channel = f"s3://{BUCKET_NAME}/{prefix_data}/testing/"

    annotation_channel = f"s3://{BUCKET_NAME}/{prefix_data}/annotations/"

    classes = [
        "wheat",
    ]
    # Container configuration

    container_name = "sagemaker-d2-train-sku110k"
    container_version = "latest"
    training_image_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{container_name}:{container_version}"

    # Metrics to monitor during training, each metric is scraped from container Stdout

    metrics = [
        {
            "Name": "training:loss",
            "Regex": "total_loss: ([0-9\\.]+)",
        },
        {
            "Name": "training:loss_cls",
            "Regex": "loss_cls: ([0-9\\.]+)",
        },
        {
            "Name": "training:loss_box_reg",
            "Regex": "loss_box_reg: ([0-9\\.]+)",
        },
        {
            "Name": "training:loss_rpn_cls",
            "Regex": "loss_rpn_cls: ([0-9\\.]+)",
        },
        {
            "Name": "training:loss_rpn_loc",
            "Regex": "loss_rpn_loc: ([0-9\\.]+)",
        },
        {
            "Name": "validation:loss",
            "Regex": "total_val_loss: ([0-9\\.]+)",
        },
        {
            "Name": "validation:loss_cls",
            "Regex": "val_loss_cls: ([0-9\\.]+)",
        },
        {
            "Name": "validation:loss_box_reg",
            "Regex": "val_loss_box_reg: ([0-9\\.]+)",
        },
        {
            "Name": "validation:loss_rpn_cls",
            "Regex": "val_loss_rpn_cls: ([0-9\\.]+)",
        },
        {
            "Name": "validation:loss_rpn_loc",
            "Regex": "val_loss_rpn_loc: ([0-9\\.]+)",
        },
    ]

    # Model Hyperparameters

    od_algorithm = "faster_rcnn"  # choose one in ("faster_rcnn", "retinanet")
    training_job_hp = {
        # Dataset
        "classes": json.dumps(classes),
        "dataset-name": json.dumps("global_wheat"),
        "label-name": json.dumps("wheat"),
        # Algo specs
        "model-type": json.dumps(od_algorithm),
        "backbone": json.dumps("R_101_FPN"),
        # Data loader
        "num-iter": 900,
        "log-period": 500,
        "batch-size": 4,
        "num-workers": 8,
        # Optimization
        "lr": 0.005,
        "lr-schedule": 3,
        # Faster-RCNN specific
        "num-rpn": 517,
        "bbox-head-pos-fraction": 0.2,
        "bbox-rpn-pos-fraction": 0.4,
        # Prediction specific
        "nms-thr": 0.2,
        "pred-thr": 0.1,
        "det-per-img": 300,
        # Evaluation
        "evaluation-type": "fast",
    }

    # Compile Sagemaker Training job object and start training
    prefix_model = f"{PREFIX}/training_artefacts"
    d2_estimator = Estimator(
        image_uri=training_image_uri,
        role=IAM_ROLE_NAME,
        sagemaker_session=sagemaker.Session(),
        instance_count=1,
        #     instance_type=training_instance,
        instance_type="ml.g4dn.12xlarge",
        hyperparameters=training_job_hp,
        environment={
            "BUCKET_NAME": BUCKET_NAME,
            "PREFIX": prefix_report,
            "REGION": REGION,
        },
        metric_definitions=metrics,
        output_path=f"s3://{BUCKET_NAME}/{prefix_model}",
        base_job_name=f"detectron2-{od_algorithm.replace('_', '-')}",
    )

    d2_estimator.fit(
        {
            "training": training_channel,
            "validation": validation_channel,
            "test": test_channel,
            "annotation": annotation_channel,
        },
    )

    training_job_name = d2_estimator.latest_training_job.name

    report = pd.read_csv(f"s3://{BUCKET_NAME}/{prefix_report}/reports.csv")

    metrics = pd.read_csv(f"s3://{BUCKET_NAME}/{prefix_report}/metrics.csv")

    message = (
        f"## Training Job Submission Report\n\n"
        f"Training Job name: '{training_job_name}'\n\n"
        "Model Artifacts Location:\n\n"
        f"'s3://{BUCKET_NAME}/{PREFIX}/output/{training_job_name}/output/model.tar.gz'\n\n"
        "See the Logs in a few minute at: "
        f"[CloudWatch](https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={training_job_name})\n\n"
        "If you merge this pull request the resulting endpoint will be avaible this URL:\n\n"
        f"'https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{training_job_name}/invocations'\n\n"
        f"## Training Job Performance Report\n\n"
        f"{metrics.to_markdown(index=False)}\n\n"
    )
    print(message)

    # Write metrics to file
    with open("details.txt", "w") as outfile:
        outfile.write(message)


if __name__ == "__main__":
    main()
