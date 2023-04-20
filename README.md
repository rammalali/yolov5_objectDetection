# Object Detection using YOLOv5

This project uses different architectures of YOLOv5 for object detection of Dolly and its wheels. The dataset used for this project can be found at: [Drive/Dataset](https://drive.google.com/drive/folders/1lZOZFiL8jLFaShl1D616N3HCKCvWhjwS?usp=sharing).

## Contents

- `train_test_notebook.ipynb`: This Jupyter Notebook file shows the process of data preprocessing, train, test, training and exporting models for the project. It is recommended to use this notebook on Jupyter Colab (with GPU).

- `fastAPI/runs/train/exp/weights/yolov5s.onnx - fastAPI/runs/train/exp/weights/yolov5m.onnx`: These are the exported model files, which can be used for deployment.

- `app.py`: This is the file that contains the code for deploying the model using FastAPI.

- `Dockerfile`: This file contains the code for Dockerizing the application.

## Credits

This project was created by [Ali Rammal](https://github.com/rammalali) for [inmind.academy].

We hope you find this project helpful. Please feel free to contribute or raise issues if you find any problems. Thank you! 
