from pipeline import *
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\François\Documents\GitHub\tesseract'

"""
compression = compressor(PATH_TO_OD_LABELS="../models/object_detection/labels",PATH_TO_OD_MODEL_DIR="../models/object_detection")

detections = compression.detect_objects("C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\explanation.jpg")
print(detections)
compression.visualize_detections()
"""
img = Image.open('C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\explanation.jpg')
print(pytesseract.image_to_string(img))