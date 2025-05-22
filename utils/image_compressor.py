import cv2
import numpy as np
from PIL import Image
import io


def compress_image(image, quality=85, max_size=None, format='JPEG'):
    """
    压缩图像

    参数:
        image: 输入图像(PIL Image或numpy数组)
        quality: 压缩质量(1-100)
        max_size: 最大尺寸(宽或高)，None表示不调整大小
        format: 输出格式('JPEG'或'PNG')

    返回:
        压缩后的图像字节
    """
    # 转换为PIL Image对象
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 调整大小
    if max_size is not None:
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 压缩
    output_buffer = io.BytesIO()
    image.save(output_buffer, format=format, quality=quality)
    compressed_image = output_buffer.getvalue()

    return compressed_image

