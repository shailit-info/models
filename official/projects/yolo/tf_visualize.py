import matplotlib.pyplot as plt
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.utils.object_detection import visualization_utils
import numpy as np
import tensorflow as tf
from official.core import exp_factory
from official.common import registry_imports
from official.core import task_factory
from official.projects.yolo.configs import yolov7
import cv2
#import matplotlib
#matplotlib.use('Qt5Agg')
#plt.switch_backend('Qt5Agg')

tf_ex_decoder = TfExampleDecoder()
category_index={
    1: {
        'id': 1,
        'name': '1'
       },
    2: {
        'id': 2,
        'name': '2'
       },
    3: {
        'id': 3,
        'name': '3'
       }
}
buffer_size = 20
num_of_examples = 10

def show_batch(raw_records, num_of_examples):
  plt.figure(figsize=(20, 20))
  use_normalized_coordinates=True
  min_score_thresh = 0.30
  for i, serialized_example in enumerate(raw_records):
    #plt.subplot(1, 3, i + 1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = decoded_tensors['image'].numpy().astype('uint8')
    scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
    print('bbox', decoded_tensors['groundtruth_boxes'].numpy())
    print('image : ', image.shape)
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        decoded_tensors['groundtruth_boxes'].numpy(),
        decoded_tensors['groundtruth_classes'].numpy().astype('int'),
        scores,
        category_index=category_index,
        use_normalized_coordinates=use_normalized_coordinates,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4)

    cv2.imwrite(
      str(i) + '_original_yolov7fulli_validation.PNG', image
    )
    #plt.imshow(image)
    #plt.axis('off')
    #plt.title(f'Image-{i+1}')
  #plt.show()

exp_config = exp_factory.get_exp_config('benjamin_yolov7x')
'''
raw_records = tf.data.TFRecordDataset(tf.io.gfile.glob(
    exp_config.task.train_data.input_path)).shuffle(
        buffer_size=buffer_size).take(num_of_examples)
'''
raw_records = tf.data.TFRecordDataset(tf.io.gfile.glob(
    exp_config.task.validation_data.input_path)).shuffle(
        buffer_size=buffer_size).take(num_of_examples)
show_batch(raw_records.shuffle(buffer_size=10), num_of_examples)
