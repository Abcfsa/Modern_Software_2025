import streamlit as sl
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.other_util import *
from utils.pdf_image import *
import tempfile
import os

sl.header("Software Project")

def redo():
    sl.session_state['pdf_image']=None

if 'allow_anno' not in sl.session_state:
    sl.session_state['allow_anno']=True
if 'pdf_image' not in sl.session_state:
    sl.session_state['pdf_image']=None

file=sl.file_uploader("PDF",type=['pdf'],on_change=refresh)

col_1,col_2=sl.columns(2)

if file is None:
    sl.write("请上传一个 PDF 文件")
else:
    # 检查文件类型
    file_format = file.name.split(".")[-1].lower()
    if file_format != "pdf":
        sl.error("请上传 PDF 格式的文件")
    else:
        # 创建临时文件保存上传的 PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file.getbuffer())
            pdf_path = f.name
        
        # 设置 DPI (图像质量)
        dpi = sl.sidebar.slider("图像质量 (DPI)", min_value=72, max_value=600, value=300, step=36)
        
        # 转换 PDF 为图像
        if sl.session_state['pdf_image']==None:
            if sl.button("转换为图片", key="convert_pdf"):
                with sl.spinner("正在处理 PDF..."):
                    sl.session_state['pdf_image'] = pdf_to_images(pdf_path, dpi=dpi)
                sl.rerun()
        else:
            sl.success(f"成功转换 {len(sl.session_state['pdf_image'])} 页")
            # 显示每一页并提供下载选项
            pg_num=sl.slider("Page_num",1,len(sl.session_state['pdf_image']),key="pg_num")
            sl.subheader(f"第 {pg_num} 页")
            sl.image(sl.session_state['pdf_image'][pg_num-1])
            _,encoded_image = cv2.imencode(".png",np.array(sl.session_state['pdf_image'][pg_num-1])[:,:,::-1])
            n_col1,n_col2=sl.columns(2)
            n_col1.download_button(
                label=f"下载第 {pg_num} 页",
                data=encoded_image.tobytes(),
                file_name=f"page_{pg_num}.png",
                mime="image/png"
            )
            n_col2.button("重做",key="Redo",on_click=redo)
            # 保存为临时文件以便下载
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_file:
                #     img.save(img_file.name, format="PNG")
                #     img_file.flush()
                    
                #     # 提供下载按钮
                #     with open(img_file.name, "rb") as f_download:
                #         sl.download_button(
                #             label=f"下载第 {i+1} 页",
                #             data=f_download,
                #             file_name=f"page_{i+1}.png",
                #             mime="image/png"
                #         )
                        
                #         # 清理临时文件
                #         os.unlink(img_file.name)
            
            # 清理 PDF 临时文件
            os.unlink(pdf_path)