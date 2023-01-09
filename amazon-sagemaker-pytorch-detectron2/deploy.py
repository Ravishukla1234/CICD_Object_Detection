#!/usr/bin/env python

import os

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorchModel

import pandas as pd
import boto3
import botocore


def main():

    # Replace with your desired configuration
    initial_instance_count = 1
    endpoint_instance_type = "ml.m5.large"

    BUCKET_NAME = os.environ["BUCKET_NAME"]
    PREFIX = os.environ["PREFIX"]
    account = os.environ["ACCOUNT_ID"]
    region = os.environ["REGION"]
    role = os.environ["ROLE"]
    serve_container_name = os.environ["SERVE_CONTAINER_NAME"]
    OBJECT_KEY_REPORT = f"{PREFIX}/reports.csv"
    OBJECT_KEY_METRICS = f"{PREFIX}/metrics.csv"

    serve_container_version = "latest"

    serve_image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{serve_container_name}:{serve_container_version}"

    s3 = boto3.resource("s3")

    try:
        s3.Bucket(BUCKET_NAME).download_file(OBJECT_KEY_REPORT, "reports.csv")

        # Load reports df
        reports_df = pd.read_csv("reports.csv")
        print(reports_df)

    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("Report.csv not found!")
        else:
            raise

    reports_df["date_time"] = pd.to_datetime(
        reports_df["date_time"], format="%Y-%m-%d %H:%M:%S"
    )
    latest_training_job_name = reports_df.sort_values(
        ["date_time"], ascending=False
    ).training_job_name.values[0]

    attached_estimator = Estimator.attach(latest_training_job_name)
    training_job_artifact = attached_estimator.model_data

    sm_session = sagemaker.Session()

    model = PyTorchModel(
        name=latest_training_job_name,
        model_data=training_job_artifact,
        role=role,
        sagemaker_session=sm_session,
        entry_point="predict_sku110k.py",
        source_dir="container_serving",
        image_uri=serve_image_uri,
        framework_version="1.6.0",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.p2.xlarge",
        tags=[{"Key": "email", "Value": "youremail@domain.com"}],
        wait=False,
    )

    print(predictor.endpoint_name)


if __name__ == "__main__":
    main()
