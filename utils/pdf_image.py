from pdf2image import convert_from_path  # 新增导入
import streamlit as sl
def pdf_to_images(pdf_path, dpi=300, fmt="PNG"):
    """
    将PDF文件转换为图片列表
    :param pdf_path: PDF文件路径
    :param dpi: 输出图片分辨率（默认300dpi）
    :param fmt: 图片格式（支持PNG/JPEG/BMP等，默认PNG）
    :return: 图片列表（每个元素为PIL.Image对象）
    """
    try:
        # 调用pdf2image转换PDF页面为图片
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt=fmt.upper(),
            single_file=False,  # 多页PDF拆分为多张图片
            use_cropbox=True    # 使用PDF的裁切框
        )
        return images
    except Exception as e:
        sl.error(f"PDF转换失败：{str(e)}")
        return None