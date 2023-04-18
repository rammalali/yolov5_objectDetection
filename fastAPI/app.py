from typing import Union
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import json
import base64
import onnxruntime
import shutil
import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from typing import Any
import uuid

from detect import run


# load the model
model_path = "runs/train/exp/weights/best.onnx"
session = onnxruntime.InferenceSession(model_path)


app = FastAPI()


class my_json(BaseModel):
    label: Any
    # label_path: str


class BASE64_input(BaseModel):
    image_path: str


def get_label_path(image_path, format="json"):
    base_path = os.path.dirname(os.path.dirname(image_path))
    image_filename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_filename)
    label_filename = f"{image_name}.{format}"
    label_path = os.path.join(base_path, "labels", format, label_filename)
    return label_path


def remove_dir(dir_path="runs/detect/exp"):
    # Remove the directory (if it's empty)
    dir_path = "runs/detect/exp"
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Remove the directory and its contents
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")
    else:
        print(f"Directory {dir_path} does not exist.")


def extract_coordinates(self, json_path):
    with open(json_path, "rb") as f:
        data = json.load(f)
    coordinates = []
    for obj in data:
        if (
            "ObjectClassName" in obj
            and "Left" in obj
            and "Top" in obj
            and "Right" in obj
            and "Bottom" in obj
        ):
            objectClassName = obj["ObjectClassName"]
            left = obj["Left"]
            top = obj["Top"]
            right = obj["Right"]
            bottom = obj["Bottom"]
            coordinates.append(
                {
                    "ObjectClassName": objectClassName,
                    "Left": left,
                    "Top": top,
                    "Right": right,
                    "Bottom": bottom,
                }
            )
    return coordinates


def base64_to_image(base64_str, save_path="images_from_base64\image.png", format="PNG"):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.save(save_path, format=format)
    return img


def txt_to_json(txt):
    # this to convert it back from txt to json
    # x_center = (left + right) / (2 * image_width)
    # y_center = (top + bottom) / (2 * image_height)
    # width = (right - left) / image_width
    # height = (bottom - top) / image_height

    image_width = 1280
    image_height = 720

    lines = txt.strip().split("\n")
    print(lines)
    json_objects = []

    for i, line in enumerate(lines):
        class_id, x_center, y_center, width, height, acc = map(float, line.split())

        left = (x_center - width / 2) * image_width
        right = (x_center + width / 2) * image_width
        top = (y_center - height / 2) * image_height
        bottom = (y_center + height / 2) * image_height

        object_class_name = "Wheel" if class_id == 1 else "Dolly"

        json_object = {
            "Id": i,
            "ObjectClassName": object_class_name,
            "ObjectClassId": int(class_id),
            "Left": round(left),
            "Top": round(top),
            "Right": round(right),
            "Bottom": round(bottom),
            "Accuracy": acc,
        }

        json_objects.append(json_object)

    json_output = json.dumps(json_objects, indent=4)
    return json_output


@app.get("/models")
def list_models():
    return {"models": ["BMW_1"]}


@app.get("/labels/{model_name}")
def get_labels(model_name: str):
    if model_name == "BMW_1":
        return {"labels": ["Dolly", "Wheel"]}
    else:
        return JSONResponse(status_code=404, content={"detail": "Model not found"})


@app.post("/{model_name}/json/", response_model=my_json)
def get_json(item: BASE64_input, model_name: str):
    remove_dir("runs/detect/exp")

    base64_to_image(item.image_path)
    # os.rmtree("runs/detect/exp5")
    if model_name == "yolov5":
        run(
            weights="runs/train/exp/weights/best.onnx",
            conf_thres=0.25,
            source="images_from_base64",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    elif model_name == "mobilenet":
        run(
            weights="runs/train/exp/weights/best.onnx",
            conf_thres=0.25,
            source="images_from_base64",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")

    with open(r"runs/detect/exp/labels/image.txt", "r") as f:
        txt = f.read()

    json_str = txt_to_json(txt)
    label = json.loads(json_str)
    remove_dir("runs/detect/exp")
    return {"label": label}


@app.post("/{model_name}/predict-box/")
async def predict_box(model_name: str, file: UploadFile = File(...)):  # async for await
    file.filename = f"image.png"
    contents = await file.read()

    # Remove the directory (if it's empty)
    remove_dir("runs/detect/exp")

    # Save the image to disk
    with open(f"data/images/{file.filename}", "wb") as f:
        f.write(contents)

    if model_name == "yolov5":
        # os.rmtree("runs/detect/exp5")
        run(
            weights="runs/train/exp/weights/best.onnx",
            conf_thres=0.25,
            source="data/images",
            save_txt=True,
            save_conf=True,
            name="exp",
        )
    elif model_name == "mobilenet":
        run(
            weights="runs/train/exp/weights/best.onnx",
            conf_thres=0.25,
            source="data/images",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")

    return FileResponse("runs/detect/exp/image.png")


@app.post("/{model_name}/predict-json/", response_model=my_json)
async def predict_json(
    model_name: str, file: UploadFile = File(...)
):  # async for await
    file.filename = f"image.png"
    contents = await file.read()

    # Remove the directory (if it's empty)
    remove_dir("runs/detect/exp")

    # Save the image to disk
    with open(f"data/images/{file.filename}", "wb") as f:
        f.write(contents)

    if model_name == "yolov5":
        # os.rmtree("runs/detect/exp5")
        run(
            weights="runs/train/exp/weights/best.onnx",
            conf_thres=0.25,
            source="data/images",
            save_txt=True,
            save_conf=True,
            name="exp",
        )
    elif model_name == "mobilenet":
        run(
            weights="runs/train/exp/weights/best.onnx",
            conf_thres=0.25,
            source="data/images",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")
    

    with open(r"runs/detect/exp/labels/image.txt", "r") as f:
        txt = f.read()

    json_str = txt_to_json(txt)
    label = json.loads(json_str)

    remove_dir("runs/detect/exp")
    return {"label": label}
