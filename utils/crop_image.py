

def crop_image(image, crop_coords):
    """
    根据用户框选的坐标裁切图片
    
    参数:
    image (PIL.Image): 原始图片
    crop_coords (dict): 包含裁切坐标的字典，格式为 {'left': x1, 'top': y1, 'right': x2, 'bottom': y2}
    
    返回:
    PIL.Image: 裁切后的图片
    """
    if not crop_coords or not all(key in crop_coords for key in ['left', 'top', 'right', 'bottom']):
        return image
    
    # 确保坐标有效
    left = max(0, min(crop_coords['left'], crop_coords['right']))
    top = max(0, min(crop_coords['top'], crop_coords['bottom']))
    right = min(image.width, max(crop_coords['left'], crop_coords['right']))
    bottom = min(image.height, max(crop_coords['top'], crop_coords['bottom']))
    
    # 裁切图片
    cropped_img = image.crop((left, top, right, bottom))
    return cropped_img