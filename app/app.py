#app.py
import os
import requests
import json
import cv2
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import ast 

import requests
import boto3
from flask import Flask, render_template, request, redirect, url_for

COLORS = [
    (100, 200, 0),
]    
s3 = boto3.resource('s3',
    aws_access_key_id=<<YOUR_AWS_ACCESS_KEY_ID>>,
    aws_secret_access_key=<<YOUR_AWS_SECRET_ACCESS_KEY>>)
PREFIX = f"detectron2/api"
BUCKET_NAME = "<<YOUR_BUCKET_NAME>>"
app = Flask(__name__)
UPLOAD_FOLDER = f"app\static"


def download_predictions_on_image(
    file: str, p_preds, score_thr: float = 0.5, show=False
) -> plt.Figure:
    r"""Plot bounding boxes predicted by an inference job on the corresponding image

    Parameters
    ----------
    p_img : np.ndarray
        input image used for prediction
    p_preds : Mapping
        dictionary with bounding boxes, predicted classes and confidence scores
    score_thr : float, optional
        show bounding boxes whose confidence score is bigger than `score_thr`, by default 0.5
    show : bool, optional
        show figure if True do not otherwise, by default True

    Returns
    -------
    plt.Figure
        figure handler

    Raises
    ------
    IOError
        If the prediction dictionary `p_preds` does not contain one of the required keys:
        `pred_classes`, `pred_boxes` and `scores`
    """


    p_img = mpimg.imread(file)
    
    for required_key in ("pred_classes", "pred_boxes", "scores"):
        if required_key not in p_preds:
            raise IOError(f"Missing required key: {required_key}")

    fig, fig_axis = plt.subplots(1)
    fig_axis.imshow(p_img)
    for class_id, bbox, score in zip(
        p_preds["pred_classes"], p_preds["pred_boxes"], p_preds["scores"]
    ):
        if score < score_thr:
            break  # bounding boxes are sorted by confidence score in descending order
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=3,
            edgecolor=[float(val) / 255 for val in COLORS[class_id]],
            facecolor="none",
        )
        fig_axis.add_patch(rect)
    plt.axis("off")
    fig_axis.get_xaxis().set_visible(False)
    fig_axis.get_yaxis().set_visible(False)
    file_name = file.split('\\')[-1]
    file_name = f"static\out_{file_name}"
    fig.savefig(file_name, bbox_inches='tight', pad_inches = 0,dpi = 500)
    if show:
        plt.show()
    return file_name

def predict(image_path,url ):
    print(f"image_path - {image_path}")
    file_name = image_path.split("\\")[-1]
    key_prefix = f"{PREFIX}/{file_name}"
    s3.Bucket(BUCKET_NAME).upload_file(image_path, key_prefix)
    s3_path = f"s3://{BUCKET_NAME}/{key_prefix}"


    body = json.dumps({
    "s3_path": s3_path
      }    )
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=body)
    detections = json.loads(response.text)["body"]
    detections = ast.literal_eval(detections)
    path = download_predictions_on_image(image_path,detections)
    return path


@app.route("/",methods = ["GET", "POST"])
def upload_predict():
	url = "<<Your API Gateway URL>>"
	if(request.method == "POST"):
		image_file = request.files["image"]
		if(image_file ):
			#image_location = f"{UPLOAD_FOLDER}\{image_file.filename}"
			image_location = f"static\{image_file.filename}"
			image_file.save(image_location)
			path = predict(image_location,url )
			return 	render_template("output.html",prediction = path, image_loc = image_file.filename)	
	return render_template("index.html")

if __name__ =="__main__":
    app.run(port = 12000, debug = True)
