from pipeline import *

compressor = compressor(PATH_TO_OD_LABELS="C:\\Users\\samze\\Documents\\GitHub\\aicompression\\models\\object_detection\\labels",PATH_TO_OD_MODEL_DIR="C:\\Users\\samze\\Documents\\GitHub\\aicompression\\models\\object_detection")

detections = compressor.detect_objects("C:\\Users\\samze\\Pictures\\Reduction.jpg")
print(detections)
