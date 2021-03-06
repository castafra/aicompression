from pipeline import *
from PIL import Image


#compression = compressor(PATH_TO_OD_LABELS="../models/object_detection/labels",PATH_TO_OD_MODEL_DIR="../models/object_detection")

#detections = compression.detect_objects("C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\slide_4.jpg")

#compression.visualize_detections()

#compression.background_retrieve()

#texts = compression.perform_ocr()
#print(texts)
#img = Image.open(r'C:\Users\François\Desktop\text_sample.jpg')
#print(pytesseract.image_to_string(img))


if __name__ == '__main__':
    #compare_image = CompareImage('C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\slide_4.jpg', 'C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\slide_4.jpg')
    #image_difference = compare_image.compare_image()
    #print(image_difference)
    
    compression = compressor(PATH_TO_OD_LABELS="../models/object_detection/labels",PATH_TO_OD_MODEL_DIR="../models/object_detection")
    detections = compression.detect_objects("C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\slide_9593.jpg")
    viz = compression.visualize_detections()
    plt.figure()
    plt.imshow(viz)
    plt.show()
    """
    texts = compression.perform_ocr()
    detected_texts = compression.detected_texts
    for k in range(len(detected_texts)):
        box = detected_texts[k]["box"]
        print(box)
        text = detected_texts[k]["text"]
        print(text)
    """
    
    #img = Image.open(r'C:\Users\François\Desktop\text_sample.jpg')
    #print(pytesseract.image_to_string(img))



