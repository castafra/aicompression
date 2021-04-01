import time
import matplotlib
import warnings
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import tensorflow as tf
import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
matplotlib.use('tkagg')


class compressor():
    def __init__(self, PATH_TO_OD_MODEL_DIR="../training_od/exported-models/model_generatedslides_V1", PATH_TO_OD_LABELS="../training_od/annotations") -> None:
        self.configs = config_util.get_configs_from_pipeline_file(
            PATH_TO_OD_MODEL_DIR + "/pipeline.config")
        self.model_config = self.configs['model']
        self.detection_model = model_builder.build(
            model_config=self.model_config, is_training=False)

        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(PATH_TO_OD_MODEL_DIR +
                          "/checkpoint", 'ckpt-0')).expect_partial()

        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_OD_LABELS+"/label_map.pbtxt",
                                                                                 use_display_name=True)

    def detect_fn(self, image):
        """Detect objects in image."""

        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        
        return detections

    def load_image_into_numpy_array(self, path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
        path: the file path to the image

        Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
        """
        self.img_path = path
        img = Image.open(path)
        return np.array(img)

    def detect_objects(self, image_path):
        """Returns the objects detected in a given image

        Args:
        image_path: the file path to the image

        Returns:
        a dataframe with the detected boxes and the corresponding classes
        """
        image_np = self.load_image_into_numpy_array(image_path)
        self.image_np = image_np
        image_shape = image_np.shape
        self.img_shape = image_shape
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)
        self.detections = detections

        detected_boxes_raw = detections['detection_boxes'][detections['detection_scores'] > 0.45]
        dims = np.diag([image_shape[0], image_shape[1],
                       image_shape[0], image_shape[1]])
        detected_boxes = list(np.dot(detected_boxes_raw, dims).astype(int))
        detected_classes_ids = detections['detection_classes'][detections['detection_scores'] > 0.45]+1
        detected_classes = []
        for k in range(len(detected_classes_ids)):
            detected_classes.append(
                self.category_index[detected_classes_ids[k]]['name'])

        detected_objects = pd.DataFrame(
            {'box': detected_boxes, 'class': detected_classes})
        print(detected_objects)
        self.detected_objects = detected_objects
        return detected_objects
    
    def visualize_detections(self):
        """
        Return a matplotlib image of the detections for the last image to which
        the object detection has been performed
        """
        detections = self.detections
        label_id_offset = 1
        image_np_with_detections = self.image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=10,
                    min_score_thresh=0.2,
                    agnostic_mode=False)

        plt.figure()
        plt.imshow(image_np_with_detections)
        plt.show()

    def perform_ocr(self):
        """
        Perform OCR (Optical Image Recognition) on the region of an image
        Args : 
        image : numpy array image on which to perform OCR
        box : list [y_min,x_min,y_max,x_max] representing the region of the image where we want to detect text
        """
        print(self.img_shape)
        text_boxes = self.detected_objects[self.detected_objects["class"] == "text"].reset_index()
        image = Image.open(self.img_path)
        for i in range(len(text_boxes)):
            box = text_boxes.box[i] 
            box[0] = max(0,box[0] - (box[2]-box[0])/10)
            box[1] = max(0,box[1] - (box[3]-box[1])/10)
            box[2] = min(self.img_shape[0],box[2] + (box[2]-box[0])/10)
            box[3] = min(self.img_shape[1],box[3] + (box[3]-box[1])/10)
            box_crop = (box[1],box[0],box[3],box[2])
            img_cropped = image.crop(box_crop)
            text = pytesseract.image_to_string(img_cropped)
            print(text)

    def background_retrieve(self):
        """Returns the retrieved background based on the objects detected

        Args:
        detected_objects: dataframe of objects boxes and their classes

        Returns:
        a PIL image of the retrieved slide background
        """
        image = Image.open(self.img_path)
        draw_slide = ImageDraw.Draw(image)
        for i in range(len(self.detected_objects)):
            r, g, b = image.getpixel(((int(self.detected_objects['box'].loc[i][1] + self.detected_objects['box'].loc[i][3]) // 2),
                                      int(self.detected_objects['box'].loc[i][0] - 1)))
            draw_slide.rectangle(
                [(self.detected_objects['box'].loc[i][1], self.detected_objects['box'].loc[i][0]),
                 (self.detected_objects['box'].loc[i][3], self.detected_objects['box'].loc[i][2])], (r, g, b))
        image.show()
        return image
