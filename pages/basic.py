import streamlit as sl
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.other_util import *
from utils.crop_image import *
from utils.image_processing_utils import *
import streamlit_drawable_canvas
from io import BytesIO 

sl.header("Software Project")

if 'allow_anno' not in sl.session_state:
    sl.session_state['allow_anno']=True

file=sl.file_uploader("Image",type=['jpg','jpeg','png'],on_change=refresh)
mode=sl.sidebar.selectbox("Select mode",["Crop","Rotate", "Draw", "Add Text"])
col_1,col_2=sl.columns(2)

if not file:
    sl.write("Please upload an image.")
else:
    file_format=file.name.split(".")[-1]
    if mode=='Crop':
        col_1.image(file)  # 显示原始图片
        sl.session_state['allow_anno'] = True  # 允许标注（用于框选裁切区域）
        
        # 使用Streamlit的标注工具实现框选
        if sl.session_state['allow_anno']:
            # 初始化标注配置（仅显示矩形框工具）
            drawing_mode = "rect"  # 固定为矩形框选模式
            stroke_width = 10
            pic = Image.open(file).convert("RGB")
            
            # 使用streamlit_drawable_canvas创建可绘制区域
            canvas_result = streamlit_drawable_canvas.st_canvas(
                fill_color="rgba(0, 255, 0, 0.3)",  # 绿色半透明填充
                stroke_color="rgba(0, 255, 0, 1)",    # 绿色边框
                stroke_width=stroke_width,
                drawing_mode=drawing_mode,
                background_image=pic,
                width=min(pic.width, 600),
                height=min(pic.height, 600),
                key="crop_canvas"
            )
            
            # 解析标注结果（获取矩形框坐标）
            if canvas_result.json_data.get("objects"):
                # sl.write(canvas_result.json_data["objects"][0])
                # 提取第一个矩形框（假设用户仅框选一个区域）
                bbox = canvas_result.json_data["objects"][0]
                xmin, ymin, xmax, ymax = bbox["left"], bbox["top"], bbox["left"] + bbox["width"], bbox["top"] + bbox["height"]
                
                # 显示裁切预览按钮
                if sl.button("Crop and Preview", key="crop_preview"):
                    # 执行裁切
                    cropped_img = crop_image(pic, {'left': xmin, 'top': ymin, 'right': xmax, 'bottom': ymax})
                    # 存储结果到会话状态
                    sl.session_state["cropped_img"] = cropped_img
                    sl.session_state['allow_anno'] = False  # 关闭标注模式
                    
        # 显示裁切结果
        if "cropped_img" in sl.session_state:
            col_2.image(sl.session_state["cropped_img"])  # 显示裁切后的图片
            # 转换为OpenCV格式以便保存
            cropped_np = np.array(sl.session_state["cropped_img"])[:, :, ::-1]
            _, encoded_image = cv2.imencode(f".{file_format}", cropped_np)
            # 提供下载按钮
            col_2.download_button(
                label="Download Cropped Image",
                data=encoded_image.tobytes(),
                file_name=f"cropped_{file.name}",
                mime=f"image/{file_format}",
                key="download_crop"
            )
            # 返回按钮
            if col_2.button("Return to Crop", key="return_crop"):
                sl.session_state.pop("cropped_img", None)
                sl.session_state['allow_anno'] = True
                sl.rerun()
    elif mode=='Rotate':
        sl.sidebar.markdown("### Image Rotation Settings")
        angle = sl.sidebar.slider("Select rotation angle (counter-clockwise):", -180.0, 180.0, 0.0, 0.5, key="rotate_angle")

        col_1.image(file, caption="Original Image", use_column_width=True)

        if sl.button("Execute Rotation", key="pro_rotate"):
            try:
                image_pil = Image.open(file)
                # For some image formats (like some PNGs), converting directly to RGBA might be better to preserve transparency
                if image_pil.mode not in ['RGB', 'RGBA']:
                        image_pil = image_pil.convert('RGBA')
                elif image_pil.mode == 'P': # Handle palette mode
                        image_pil = image_pil.convert('RGBA')

                rotated_image_pil = rotate_image_pil(image_pil, angle)
                sl.session_state['processed_image_display'] = rotated_image_pil
                sl.session_state['processed_image_format'] = file.name.split(".")[-1]
                sl.session_state['processed_image_action'] = "rotate" # For download filename and state management
            except Exception as e:
                sl.error(f"Error during image rotation: {e}")
                if 'processed_image_display' in sl.session_state:
                    del sl.session_state['processed_image_display']

        if sl.session_state.get('processed_image_action') == "rotate" and 'processed_image_display' in sl.session_state:
            img_to_display = sl.session_state['processed_image_display']
            col_2.image(img_to_display, caption=f"Rotated Image ({angle}°)", use_column_width=True)

            # Prepare for download
            buf = BytesIO()
            # Rotated image might have transparent areas, PNG is a good format
            save_fmt = 'PNG' if img_to_display.mode == 'RGBA' else sl.session_state['processed_image_format'].upper()
            if save_fmt.lower() == 'jpg': save_fmt = 'JPEG' # Pillow needs 'JPEG'
            
            try:
                # If the original image was JPEG and had no Alpha channel, the rotated one might still be RGB and can be saved as JPEG
                if img_to_display.mode == 'RGB' and save_fmt == 'JPEG':
                    img_to_display.save(buf, format=save_fmt, quality=95)
                else: # For other cases, especially those with an Alpha channel, save as PNG
                    save_fmt = 'PNG' # Force PNG to preserve transparency
                    img_to_display.save(buf, format=save_fmt)
                
                byte_im = buf.getvalue()
                col_2.download_button(
                    label=f"Download Rotated Image (. {save_fmt.lower()})",
                    data=byte_im,
                    file_name=f"rotated_{file.name.split('.')[0]}.{save_fmt.lower()}",
                    mime=f"image/{save_fmt.lower()}",
                    key="down_rotate"
                )
            except Exception as e:
                col_2.error(f"Error preparing download: {e}")
            
            if col_2.button("Clear Rotation Result", key="clear_rotate_res"):
                del sl.session_state['processed_image_display']
                if 'processed_image_action' in sl.session_state:
                    del sl.session_state['processed_image_action']
                sl.rerun()
    elif mode == "Draw":
        sl.sidebar.markdown("### Drawing Settings")
        drawing_tool = sl.sidebar.selectbox(
            "Drawing Tool:",
            ("freedraw", "line", "rect", "circle", "transform", "point"),
            index=0,
            key="drawing_tool_draw"
        )
        stroke_width_val = sl.sidebar.slider("Stroke Width: ", 1, 50, 5, key="stroke_width_draw")
        hex_stroke_color = sl.sidebar.color_picker("Stroke Color:", "#FF0000", key="stroke_color_draw_hex")

        try:
            # Load the original image. This will be used for the canvas background
            # and as the base for the final composited image.
            pil_image_for_canvas_and_composite = Image.open(file)
        except Exception as e:
            sl.error(f"Failed to load background image: {e}")
            # Provide a placeholder if image loading fails
            pil_image_for_canvas_and_composite = Image.new("RGBA", (600, 400), (200, 200, 200, 255))

        # Dynamically adjust canvas dimensions based on the image, limiting max width
        CANVAS_MAX_WIDTH = 600
        img_w, img_h = pil_image_for_canvas_and_composite.size
        aspect_ratio = img_h / img_w if img_w > 0 else 1 # Avoid division by zero
        canvas_width = min(img_w, CANVAS_MAX_WIDTH)
        # Ensure canvas_width is not zero if img_w is very small but not zero
        if canvas_width == 0 and img_w > 0: canvas_width = img_w

        canvas_height = int(canvas_width * aspect_ratio)
        if canvas_height == 0 and img_h > 0: # Handle cases where aspect_ratio might lead to 0 height
            canvas_height = int(CANVAS_MAX_WIDTH * (img_h / img_w if img_w > 0 else 1))
            if canvas_height == 0: canvas_height = img_h # Fallback to original height if still 0

        # Ensure minimum canvas size if calculated dimensions are too small or zero
        if canvas_width <=0 : canvas_width = 300 # Min default width
        if canvas_height <=0 : canvas_height = 200 # Min default height


        sl.write("Draw on the canvas below (based on uploaded image):")
        # The background_image is passed to the canvas for display during drawing.
        # The component scales this image visually to fit canvas_width and canvas_height.
        canvas_result = streamlit_drawable_canvas.st_canvas(
            fill_color="rgba(255, 165, 0, 0.0)",  # Example: shapes filled transparently
            stroke_width=stroke_width_val,
            stroke_color=hex_stroke_color,
            background_image=pil_image_for_canvas_and_composite,
            drawing_mode=drawing_tool,
            width=canvas_width,
            height=canvas_height,
            update_streamlit=True,
            key="canvas_draw_mode",
        )

        # col_1 is used for immediate preview of what the canvas returns
        col_1.write("Current Canvas Preview (raw output):")
        if canvas_result.image_data is not None:
            col_1.image(canvas_result.image_data, caption="Raw Canvas Output Preview", use_column_width=True)
        
        if sl.button("Confirm and Process Drawing", key="process_draw_button"):
            if canvas_result.image_data is not None and \
                canvas_result.image_data.shape[0] > 0 and \
                canvas_result.image_data.shape[1] > 0:
                
                # canvas_result.image_data is assumed to be the drawing layer (strokes)
                # on a transparent background, with dimensions (canvas_height, canvas_width, 4 for RGBA).
                drawing_layer_pil = Image.fromarray(canvas_result.image_data.astype(np.uint8), 'RGBA')

                # Prepare the base image for compositing:
                # 1. Take the original image.
                # 2. Resize it to the dimensions of the canvas (canvas_width, canvas_height),
                #    because the drawing layer corresponds to these dimensions.
                # 3. Convert it to RGBA format for alpha compositing.
                base_image_resized_pil = pil_image_for_canvas_and_composite.resize(
                    (canvas_width, canvas_height), 
                    Image.LANCZOS # Use a high-quality resampling filter
                )
                base_image_rgba = base_image_resized_pil.convert("RGBA")
                
                # Alpha composite the drawing layer onto the (resized) base image.
                # The base image is the first argument, the overlay (drawing) is the second.
                final_image_pil = Image.alpha_composite(base_image_rgba, drawing_layer_pil)
                
                sl.session_state['processed_image_display'] = final_image_pil
                sl.session_state['processed_image_format'] = "png" # PNG supports transparency
                sl.session_state['processed_image_action'] = "draw"
            else:
                sl.warning("Canvas data is empty or invalid. Please draw something on the image.")

        # This part (displaying in col_2 and download) should now show the composited image
        if sl.session_state.get('processed_image_action') == "draw" and 'processed_image_display' in sl.session_state:
            img_to_display = sl.session_state['processed_image_display']
            col_2.image(img_to_display, caption="Final Drawing Result", use_column_width=True)
            
            buf = BytesIO()
            save_fmt = 'PNG' 
            try:
                img_to_display.save(buf, format=save_fmt)
                byte_im = buf.getvalue()
                col_2.download_button(
                    label=f"Download Drawing (. {save_fmt.lower()})",
                    data=byte_im,
                    file_name=f"drawn_{file.name.split('.')[0]}.{save_fmt.lower()}",
                    mime=f"image/{save_fmt.lower()}",
                    key="down_draw"
                )
            except Exception as e:
                    col_2.error(f"Error preparing download: {e}")

            if col_2.button("Clear Drawing Result", key="clear_draw_res"):
                del sl.session_state['processed_image_display']
                if 'processed_image_action' in sl.session_state:
                    del sl.session_state['processed_image_action']
                sl.rerun()
    elif mode == "Add Text":
        sl.sidebar.markdown("### Add Text Settings")
        text_content = sl.sidebar.text_input("Enter text:", "Streamlit!", key="text_input_add")
        font_size_val = sl.sidebar.slider("Font Size:", 10, 200, 30, key="font_size_add")
        
        hex_text_color = sl.sidebar.color_picker("Text Color:", "#FFFFFF", key="text_color_add") # Default white
        # Convert HEX to RGBA tuple (Pillow needs it)
        r_text = int(hex_text_color[1:3], 16)
        g_text = int(hex_text_color[3:5], 16)
        b_text = int(hex_text_color[5:7], 16)
        text_color_rgba = (r_text, g_text, b_text, 255) # A=255 means opaque

        # Text position (top-left)
        # Try to get reasonable defaults or ranges from image dimensions
        try:
            temp_img_for_dims = Image.open(file)
            max_x = temp_img_for_dims.width - 10
            max_y = temp_img_for_dims.height - 10
        except:
            max_x = 500
            max_y = 500

        pos_x_val = sl.sidebar.number_input("X Coordinate:", 0, max_x, 10, key="pos_x_add")
        pos_y_val = sl.sidebar.number_input("Y Coordinate:", 0, max_y, 10, key="pos_y_add")
        text_position = (pos_x_val, pos_y_val)

        # Display font loading status
        sl.sidebar.caption(get_font_load_status_message())

        col_1.image(file, caption="Original Image", use_column_width=True)

        if sl.button("Add Text to Image", key="pro_add_text"):
            try:
                image_pil = Image.open(file)
                # Ensure the image is in RGBA mode to correctly overlay text (especially if text has transparency or image has transparent areas)
                if image_pil.mode not in ['RGB', 'RGBA']:
                        image_pil = image_pil.convert('RGBA')
                elif image_pil.mode == 'P': # Handle palette mode
                        image_pil = image_pil.convert('RGBA')

                text_added_image_pil = add_text_to_image_pil(
                    image_pil,
                    text_content,
                    text_position,
                    font_name=None, # Use DEFAULT_FONT_PATH from the module
                    font_size=font_size_val,
                    text_color=text_color_rgba
                )
                sl.session_state['processed_image_display'] = text_added_image_pil
                sl.session_state['processed_image_format'] = "png" # Images with text are best saved as PNG
                sl.session_state['processed_image_action'] = "add_text"
            except Exception as e:
                sl.error(f"Error adding text: {e}")
                if 'processed_image_display' in sl.session_state:
                    del sl.session_state['processed_image_display']

        if sl.session_state.get('processed_image_action') == "add_text" and 'processed_image_display' in sl.session_state:
            img_to_display = sl.session_state['processed_image_display']
            col_2.image(img_to_display, caption="Image with Added Text", use_column_width=True)
            
            buf = BytesIO()
            save_fmt = 'PNG' # Always save images with text as PNG for quality and transparency
            try:
                img_to_display.save(buf, format=save_fmt)
                byte_im = buf.getvalue()
                col_2.download_button(
                    label=f"Download Image with Text (. {save_fmt.lower()})",
                    data=byte_im,
                    file_name=f"text_added_{file.name.split('.')[0]}.{save_fmt.lower()}",
                    mime=f"image/{save_fmt.lower()}",
                    key="down_add_text"
                )
            except Exception as e:
                col_2.error(f"Error preparing download: {e}")

            if col_2.button("Clear Text Result", key="clear_text_res"):
                del sl.session_state['processed_image_display']
                if 'processed_image_action' in sl.session_state:
                    del sl.session_state['processed_image_action']
                sl.rerun()


