import streamlit as sl
import streamlit_authenticator as stauth
import tempfile
import shutil
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from streamlit_image_annotation import detection
import colorizers
from simple_lama_inpainting import SimpleLama
import yaml
import streamlit_drawable_canvas
from yaml.loader import SafeLoader

def refresh():
    sl.session_state['label_dict']={'bboxes':[],'labels':[]}
    sl.session_state['allow_anno']=True
    sl.session_state['rep_res']=None

def disable_choosing():
    sl.session_state['allow_anno']=True

def repair_process(pic,pre_mask):
    image_np = np.array(pic)
    pre_mask=np.array(pre_mask)
    # sl.write(image_np.shape) 
    # image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
    pre_mask=pre_mask[:,:,3]
    pre_mask=cv2.resize(pre_mask,(image_np.shape[1],image_np.shape[0]))
    pre_mask[pre_mask>0]=255
    sl.session_state['rep_res'] = np.array(sl.session_state['simplelama'](image_np,pre_mask))
    # # sl.image(result,channels='BGR')
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

with open('./auth/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    './auth/config.yaml',
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

if 'allow_login' not in sl.session_state:
    sl.session_state['allow_login']=True
if 'allow_reg' not in sl.session_state:
    sl.session_state['allow_reg']=False

# sl.sidebar.write(sl.session_state['allow_login'],sl.session_state['allow_reg'])

if sl.session_state['allow_reg']:
    try:
        email_of_registered_user, \
        username_of_registered_user, \
        name_of_registered_user = authenticator.register_user(pre_authorized=config['pre-authorized']['emails'])
        if email_of_registered_user:
            sl.success('User registered successfully')
            sl.session_state['allow_reg']=False
            sl.session_state['allow_login']=True
    except Exception as e:
        sl.error(e)
    if sl.button("Return",key="ret_but"):
        sl.session_state['allow_reg']=False
        sl.session_state['allow_login']=True
        sl.rerun()

if sl.session_state['allow_login']:
    try:
        authenticator.login()
    except Exception as e:
        sl.error(e)
    if not sl.session_state.get('authentication_status'):
        if sl.button("Register",key="reg_but"):
            sl.session_state['allow_login']=False
            sl.session_state['allow_reg']=True
            sl.rerun()

if sl.session_state.get('authentication_status'):
    with sl.sidebar:
        authenticator.logout()
        sl.write(f'Welcome *{sl.session_state.get("name")}*')

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
                # sl.write(path.shape)
                col_2.image(path,channels="BGR")
                _,encoded_image = cv2.imencode(f".{file_format}",path)
                col_2.download_button(label="Download",data=encoded_image.tobytes(),\
                                    file_name="result."+file_format,mime=f"image/{file_format}",key="down_d")
        elif mode=="Repair":
            if sl.session_state['allow_anno']:
                Drawing_mode=sl.sidebar.selectbox("Drawing tool:",
                ["freedraw", "line", "rect", "circle", "transform", "polygon", "point"],index=1)
                stroke_width = sl.sidebar.slider("Stroke width: ", 1, 25, 15)
                pic=Image.open(file).convert("RGB")
                test=streamlit_drawable_canvas.st_canvas(
                    fill_color="rgba(222, 222, 222, 0.5)",
                    stroke_color="rgba(222, 222, 222, 0.5)",
                    stroke_width=stroke_width,
                    drawing_mode=Drawing_mode,
                    background_image=pic,
                    width=min(np.array(pic).shape[1],600),
                    height=np.array(pic).shape[0]*(min(np.array(pic).shape[1],600)/np.array(pic).shape[1])
                )
                sl.button("Process",key="pro_r",on_click=repair_process,args=[pic,test.image_data])
            else:
                sl.image(sl.session_state["rep_res"])
                _,encoded_image = cv2.imencode(f".{file_format}",sl.session_state["rep_res"])
                a_col,b_col=sl.columns(2)
                a_col.download_button(label="Download",data=encoded_image.tobytes(),\
                                    file_name="result."+file_format,mime=f"image/{file_format}",key="down_r")
                if b_col.button("Return",key="rep_ret"):
                    sl.session_state['allow_anno']=True
                    sl.rerun()
                
                
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
        # elif mode=='Canvas':
        #     test=streamlit_drawable_canvas.st_canvas(
        #         stroke_color="rgba(222, 222, 222, 0.5)",
        #         background_image=Image.open(file)
        #     )
        #     sl.image(test.image_data)
            # sl.write(test.json_data)

elif sl.session_state.get('authentication_status') is False:
    sl.error('Username/password is incorrect')
elif sl.session_state.get('authentication_status') is None:
    sl.warning('Please enter your username and password')

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