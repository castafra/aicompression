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
        return np.array(Image.open(path))

    def detect_objects(self, image_path):
        """Returns the objects detected in a given image

        Args:
        image_path: the file path to the image

        Returns:
        a dataframe with the detected boxes and the corresponding classes
        """
        image_np = self.load_image_into_numpy_array(image_path)
        image_shape = image_np.shape
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

        detected_boxes_raw = detections['detection_boxes'][detections['detection_scores'] > 0.5]
        dims = np.diag([image_shape[0],image_shape[1],image_shape[0],image_shape[1]])
        detected_boxes = list(np.dot(detected_boxes_raw,dims).astype(int))
        detected_classes_ids = detections['detection_classes'][detections['detection_scores'] > 0.5]+1
        detected_classes = []
        for k in range(len(detected_classes_ids)):
            detected_classes.append(
                self.category_index[detected_classes_ids[k]]['name'])

        detected_objects = pd.DataFrame(
            {'box': detected_boxes, 'class': detected_classes})
        self.detected_objects = detected_objects
        return detected_objects

    def background_retrieve(self, detected_objects, image):
        """Returns the retrieved background based on the objects detected

        Args:
        detected_objects: dataframe of objects boxes and their classes

        Returns:
        a PIL image of the retrieved slide background
        """

        draw_slide = ImageDraw.Draw(image)
        for i in range(len(detected_objects)):
            r, g, b = image.getpixel(((detected_objects['box'].loc[i][1] + detected_objects['box'].loc[i][3]) // 2,
                                      detected_objects['box'].loc[i][0] - 1))
            draw_slide.rectangle(
                [(detected_objects['box'].loc[i][1], detected_objects['box'].loc[i][0]),
                 (detected_objects['box'].loc[i][3], detected_objects['box'].loc[i][2])], (r, g, b))
        image.show()
        return image

