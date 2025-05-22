from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Attempt to determine a default font path and provide loading information
DEFAULT_FONT_PATH = None
FONT_LOAD_STATUS_MESSAGE = "Warning: Could not find common TTF fonts (Arial, DejaVuSans). Pillow will use its built-in bitmap font, and text quality may be limited."

try:
    # Try to load Arial (common on Windows, macOS)
    ImageFont.truetype("arial.ttf", 15)
    DEFAULT_FONT_PATH = "arial.ttf"
    FONT_LOAD_STATUS_MESSAGE = "Font Hint: Will attempt to use Arial font (if available)."
except IOError:
    try:
        # Try to load DejaVuSans (common on Linux)
        ImageFont.truetype("DejaVuSans.ttf", 15)
        DEFAULT_FONT_PATH = "DejaVuSans.ttf"
        FONT_LOAD_STATUS_MESSAGE = "Font Hint: Will attempt to use DejaVuSans font (if available)."
    except IOError:
        # If both fail, DEFAULT_FONT_PATH remains None
        # FONT_LOAD_STATUS_MESSAGE is already set to the warning message
        pass

def get_font_load_status_message():
    """Returns the font loading status message to be displayed in the UI."""
    return FONT_LOAD_STATUS_MESSAGE

def rotate_image_pil(image_pil: Image.Image, angle: float) -> Image.Image:
    """
    Rotates an image using Pillow.
    Parameters:
        image_pil (PIL.Image.Image): Pillow image object.
        angle (float): Rotation angle in degrees. In Pillow, a positive value means counter-clockwise rotation.
    Returns:
        PIL.Image.Image: The rotated Pillow image object.
    """
    # Use high-quality resampling (Image.BICUBIC) and allow the image to expand to fully display the rotated content
    return image_pil.rotate(angle, resample=Image.BICUBIC, expand=True)

def add_text_to_image_pil(image_pil: Image.Image, text: str, position: tuple,
                          font_name: str = None, font_size: int = 30,
                          text_color: tuple = (255, 255, 255, 255)) -> Image.Image:
    """
    Adds text to an image.
    Parameters:
        image_pil (PIL.Image.Image): Pillow image object.
        text (str): The text to add.
        position (tuple): Top-left coordinates (x, y) for the text.
        font_name (str, optional): Font file name or path (e.g., 'arial.ttf').
                                   If None, attempts to use DEFAULT_FONT_PATH defined in the module.
        font_size (int, optional): Font size. Defaults to 30.
        text_color (tuple, optional): Text color (R, G, B, A) or (R, G, B). Defaults to white RGBA(255,255,255,255).
    Returns:
        PIL.Image.Image: The Pillow image object with added text.
    """
    # Create an editable copy of the image and ensure it's in RGBA format to support text with a transparent background
    editable_image = image_pil.copy().convert("RGBA")
    draw = ImageDraw.Draw(editable_image)

    # Determine the font path to use
    resolved_font_name = font_name if font_name else DEFAULT_FONT_PATH
    font_to_use = None

    try:
        if resolved_font_name:
            font_to_use = ImageFont.truetype(resolved_font_name, font_size)
        else:
            # If no TTF font path is available, load Pillow's default (bitmap) font
            font_to_use = ImageFont.load_default()
    except IOError:
        # If the specified font file fails to load, also fall back to the default font
        font_to_use = ImageFont.load_default()
        # The main application can notify the user about font loading failure
    except Exception: # Catch other possible font-related errors
        font_to_use = ImageFont.load_default()

    # Use anchor='lt' (left-top) to ensure the position parameter corresponds to the top-left corner of the text box
    draw.text(position, text, fill=text_color, font=font_to_use, anchor='lt')
    return editable_image

def process_drawing_canvas_output(canvas_image_data: np.ndarray) -> Image.Image:
    """
    Processes image data from streamlit_drawable_canvas (usually a NumPy array).
    Converts it to a PIL.Image object.
    Parameters:
        canvas_image_data (np.ndarray): NumPy array returned by streamlit_drawable_canvas (usually H, W, 4, uint8 RGBA).
    Returns:
        PIL.Image.Image: The converted Pillow image object, or None if the input is None.
    """
    if canvas_image_data is not None:
        # streamlit_drawable_canvas usually returns an RGBA uint8 NumPy array
        return Image.fromarray(canvas_image_data.astype(np.uint8), 'RGBA')
    return None