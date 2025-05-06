import streamlit as sl
import tempfile
import shutil
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from streamlit_image_annotation import detection
import colorizers
from simple_lama_inpainting import SimpleLama
def refresh():
    sl.session_state['label_dict']={'bboxes':[],'labels':[]}
    sl.session_state['allow_anno']=True
    sl.session_state['rep_res']=None

def disable_choosing():
    sl.session_state['allow_anno']=True

def repair_process(file,labels):
    file_bytes=file.getvalue()
    image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR) 
    # sl.write(image_np.shape) 
    # image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    for v in labels:
        (x1,y1)=(int(v['bbox'][0]),int(v['bbox'][1]))
        (x2,y2)=(int(x1+v['bbox'][2]),int(y1+v['bbox'][3]))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    sl.session_state['rep_res'] = np.array(sl.session_state['simplelama'](image_np,mask))
    # sl.image(result,channels='BGR')
    sl.session_state['allow_anno']=False

def upscale(model_path, model_name, scale, img):
    scale = int(scale.split('x')[0])
    sl.session_state['sr'].readModel(model_path)
    sl.session_state['sr'].setModel(model_name, scale)
    result = sl.session_state['sr'].upsample(img)
    return result[:, :, ::-1]

def get_modelname(model):
    if 'EDSR' in model:
        return 'edsr'
    elif 'LapSRN' in model:
        return 'lapsrn'
    elif 'ESPCN' in model:
        return 'espcn'
    elif 'FSRCNN' in model:
        return 'fsrcnn'
    elif 'LapSRN' in model:
        return 'lapsrn'

# 选择设备 (CPU 或 GPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 加载模型
if 'model' not in sl.session_state:
    sl.session_state['model']=None
if 'ecc16' not in sl.session_state:
    sl.session_state['ecc16']=colorizers.eccv16().eval().cuda()
if 'simplelama' not in sl.session_state:
    sl.session_state['simplelama']=SimpleLama()
if 'label_dict' not in sl.session_state:
    sl.session_state['label_dict']={'bboxes':[],'labels':[]}
if 'allow_anno' not in sl.session_state:
    sl.session_state['allow_anno']=True
if 'rep_res' not in sl.session_state:
    sl.session_state['rep_res']=None
if 'sr' not in sl.session_state:
    sl.session_state['sr'] = cv2.dnn_superres.DnnSuperResImpl_create()

if sl.session_state['model']==None:    
    model = YOLO('./yolo11m.pt')
    sl.session_state['model']=model
    del model


sl.header("Software Project")
file=sl.file_uploader("Image",type=['jpg','jpeg','png'],on_change=refresh)
mode=sl.sidebar.selectbox("Select mode",["Detect","Repair","Colorize",'Upscale'])
# sl.sidebar.button("Test",key='Test')

col_1,col_2=sl.columns(2)

if not file:
    sl.write("Please upload an image.")
else:
    file_format=file.name.split(".")[-1]
    # sl.write(file_format)
    if mode=="Detect":
        col_1.image(file)
        if sl.button("Process",key="pro_d"):
            file_bytes=file.getvalue()
            image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)   
            result= sl.session_state['model'].predict(image_np,device=device)
            path=result[0].plot()
            sl.write(path.shape)
            col_2.image(path,channels="BGR")
            _,encoded_image = cv2.imencode(f".{file_format}",path)
            col_2.download_button(label="Download",data=encoded_image.tobytes(),\
                                  file_name="result."+file_format,mime=f"image/{file_format}",key="down_d")
    elif mode=="Repair":
        if sl.session_state['allow_anno']:
            new_labels = detection(image_path=file, 
                                    label_list=['Watermark'],
                                    bboxes=sl.session_state['label_dict']['bboxes'],
                                    labels=sl.session_state['label_dict']['labels'],
                                    line_width=1
                                    )
            sl.info('Click Complete button after making changes to apply the change.\n\
                If not clicked,the changes will not be saved.', icon="ℹ️")
            # sl.write(new_labels)
            sl.button("Process",key="pro_r",on_click=repair_process,args=[file,new_labels])
        else:
            sl.image(sl.session_state["rep_res"],channels="BGR")
            _,encoded_image = cv2.imencode(f".{file_format}",sl.session_state["rep_res"])
            sl.download_button(label="Download",data=encoded_image.tobytes(),\
                                  file_name="result."+file_format,mime=f"image/{file_format}",key="down_r")
        # if new_labels is not None:
        #     changed=np.array_equal(sl.session_state['label_dict']['bboxes'],[v['bbox'] for v in new_labels]) and\
        #         np.array_equal(sl.session_state['label_dict']['labels'],[v['label_id'] for v in new_labels])
        #     if changed:
        #         sl.session_state['label_dict']['bboxes'] = [v['bbox'] for v in new_labels]
        #         sl.session_state['label_dict']['labels'] = [v['label_id'] for v in new_labels]
        # sl.json(sl.session_state['label_dict'])
    elif mode=="Colorize":
        col_1.image(file)
        if sl.button("Process",key="pro_c"):
            img_np=colorizers.load_img(file)
            (img_og,img_te)=colorizers.preprocess_img(img_np, HW=(256, 256))
            img_te=img_te.cuda()
            out=sl.session_state['ecc16'](img_te).cpu()
            col_img=np.array(colorizers.postprocess_tens(img_og,out)*255,np.uint8)
            col_2.image(col_img)
            _,encoded_image = cv2.imencode(f".{file_format}",col_img[:,:,::-1])
            col_2.download_button(label="Download",data=encoded_image.tobytes(),\
                                file_name="result."+file_format,mime=f"image/{file_format}",key="down_c")
    elif mode=="Upscale":
        scale=sl.sidebar.selectbox("Scale",["2x","3x","4x"],key="scale",index=None)
        if scale:
            s_model=sl.sidebar.selectbox("Model",[m.split(".")[0] for m in os.listdir(f"./models/{scale}")],\
                                       key="s_model",index=None)
            if s_model:
                col_1.image(file)
                file_bytes=file.getvalue()
                image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR) 
                if sl.button("Upscale",key="pro_s"):
                    model_path=f"./models/{scale}/{s_model}.pb"
                    model_name=get_modelname(s_model)
                    res=upscale(model_path,model_name,scale,image_np)
                    col_2.image(res)
                    _,encoded_image = cv2.imencode(f".{file_format}",res[:,:,::-1])
                    col_2.download_button(label="Download",data=encoded_image.tobytes(),\
                                        file_name="result."+file_format,mime=f"image/{file_format}",key="down_u")

# if col_1.button("Process",key="process",disabled=(file==None),on_click=disable_choosing):
#     file_bytes=file.getvalue()
#     image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)   
#     if mode=="Detect":
#         result= sl.session_state['model'].predict(image_np,device=device)
#         path=result[0].plot()
#         col_2.image(path,channels="BGR")

#     elif mode=="Repair":
#         pass
# s_col1,s_col2,s_col3=sl.columns(3)

# if s_col1.button("Detect",key="button1",disabled=(file==None)):
#     pass
    # file_bytes=file.getvalue()
    # image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    # result= sl.session_state['model'].predict(image_np,device=device)
    # path=result[0].plot()
    # col_2.image(path,channels="BGR")

# if s_col2.button("Repair",key="button2",disabled=(file==None)):
#     file_bytes=file.getvalue()
#     image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
#     img_name=r"./cur.jpg"
#     cv2.imwrite(img_name,image_np)
#     new_labels = detection(image_path=img_name, 
#                                 label_list=['Watermark'],
#                                 bboxes=[],
#                                 labels=[],
#                                 height=image_np.shape[0],
#                                 width=image_np.shape[1],
#                                 line_width=1
#                                 )
    # result= sl.session_state['model'].predict(image_np,device=device)
    # path=result[0].plot()
    # col_2.image(path,channels="BGR")