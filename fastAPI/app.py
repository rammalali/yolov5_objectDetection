from typing import Union, Any
import os
import shutil
import json
import base64
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

from detect import run

model_path = "runs/train/exp/weights/yolov5.onnx"
conf_threshold = 0.65
app = FastAPI()

class ImageJson(BaseModel):
    label: Any

class BASE64Input(BaseModel):
    image_path: str

def get_label_path(image_path, format="json"):
    base_path = os.path.dirname(os.path.dirname(image_path))
    image_filename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_filename)
    label_filename = f"{image_name}.{format}"
    label_path = os.path.join(base_path, "labels", format, label_filename)
    return label_path

def remove_dir(dir_path="runs/detect/exp"):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")
    else:
        print(f"Directory {dir_path} does not exist.")

def extract_coordinates(json_path):
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

def txt_to_json(txt, image_width=1280, image_height=720):
    lines = txt.strip().split("\n")
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
async def list_models():
    return {"models": ["yolov5", "faster-rcnn"]}


@app.get("/labels/{model_name}")
def get_labels(model_name: str):
    if model_name == "yolov5":
        return {"labels": ["Dolly", "Wheel"]}
    
    elif model_name == "faster-rcnn":
        return {"labels": ["Dolly", "Wheel"]}
    
    else:
        return JSONResponse(status_code=404, content={"detail": "Model not found"})

@app.post("/{model_name}/json/", response_model=ImageJson)
def get_json(item: BASE64Input, model_name: str):
    remove_dir("runs/detect/exp")
    base64_to_image(item.image_path)

    if model_name == "yolov5":
        run(
            weights="runs/train/exp/weights/yolov5.onnx",
            conf_thres=conf_threshold,
            source="images_from_base64",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    elif model_name == "faster-rcnn":
        return {"label": "model not done yet!"}

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")

    with open(r"runs/detect/exp/labels/image.txt", "r") as f:
        txt = f.read()

    json_str = txt_to_json(txt)
    label = json.loads(json_str)
    remove_dir("runs/detect/exp")
    return {"label": label}

@app.post("/{model_name}/predict-box/")
async def predict_box(model_name: str, file: UploadFile = File(...)):
    file.filename = f"image.png"
    contents = await file.read()
    remove_dir("runs/detect/exp")

    with open(f"data/images/{file.filename}", "wb") as f:
        f.write(contents)

    if model_name == "yolov5":
        run(
            weights="runs/train/exp/weights/yolov5.onnx",
            conf_thres=conf_threshold,
            source="data/images",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    elif model_name == "faster-rcnn":
        return {"label": "model not done yet!"}

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")

    return FileResponse("runs/detect/exp/image.png")

@app.post("/{model_name}/predict-json/", response_model=ImageJson)
async def predict_json(model_name: str, file: UploadFile = File(...)):
    file.filename = f"image.png"
    contents = await file.read()
    remove_dir("runs/detect/exp")

    with open(f"data/images/{file.filename}", "wb") as f:
        f.write(contents)

    if model_name == "yolov5":
        run(
            weights="runs/train/exp/weights/yolov5.onnx",
            conf_thres=conf_threshold,
            source="data/images",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    elif model_name == "faster-rcnn":
        return {"label": "model not done yet!"}

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")

    with open(r"runs/detect/exp/labels/image.txt", "r") as f:
        txt = f.read()

    json_str = txt_to_json(txt)
    label = json.loads(json_str)
    remove_dir("runs/detect/exp")
    return {"label": label}
