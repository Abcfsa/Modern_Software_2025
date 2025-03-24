import streamlit as sl
import tempfile
import shutil
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# 选择设备 (CPU 或 GPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 加载模型
if 'model' not in sl.session_state:
    sl.session_state['model']=None

if sl.session_state['model']==None:    
    model = YOLO('./yolo11m.pt')
    sl.session_state['model']=model
    del model


sl.header("Software Project")
file=sl.file_uploader("Image",type=['jpg','jpeg','png'])

col_1,col_2=sl.columns(2)

if not file:
    sl.write("Please upload an image.")
else:
    col_1.image(file)

if sl.button("Detect",key="button1"):
    file_bytes=file.getvalue()
    image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    result= sl.session_state['model'].predict(image_np,device=device)
    path=result[0].save()
    col_2.image(path)