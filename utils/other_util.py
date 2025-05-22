import streamlit as sl
import numpy as np
import cv2

def refresh():
    sl.session_state['pdf_image']=None
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