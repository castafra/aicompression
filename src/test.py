from pipeline import *

compressor = compressor(PATH_TO_OD_LABELS="../models/object_detection/labels",PATH_TO_OD_MODEL_DIR="../models/object_detection")

detections = compressor.detect_objects("C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\explanation.jpg")
print(detections)
