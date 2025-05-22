import streamlit as sl
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.other_util import *
from utils.image_compressor import *

sl.header("Software Project")
file=sl.file_uploader("Image",type=['jpg','jpeg','png'],on_change=refresh)
col_1,col_2=sl.columns(2)
if not file:
    sl.write("Please upload an image.")
else:
    file_format=file.name.split(".")[-1]
    col_1.image(file)

    quality = sl.sidebar.slider("压缩质量", 1, 100, 85)
    max_size = sl.sidebar.number_input("最大尺寸(像素)", min_value=32, max_value=4096, value=1024, step=32)
    format = sl.sidebar.selectbox("输出格式", ['JPEG', 'PNG'])

    if sl.button("压缩", key="pro_com"):
        file_bytes = file.getvalue()
        image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        compressed = compress_image(image_np, quality=quality, max_size=max_size, format=format)

        col_2.image(compressed)

        col_2.download_button(
            label="下载压缩图",
            data=compressed,
            file_name=f"compressed.{format.lower()}",
            mime=f"image/{format.lower()}",
            key="down_com"
        )

