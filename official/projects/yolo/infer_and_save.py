import os
from absl import app
from absl import flags
import cv2

from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
import tensorflow as tf

BUFFER_SIZE = 20
MIN_SCORE_THRESH = 0.3  # Change to see all bounding boxes confidences.

FLAGS = flags.FLAGS


flags.DEFINE_integer('num_of_examples',
                     10, 'First --num_of_examples will be saved.')
flags.DEFINE_string('tfrecords_path',
                    'gs://model_tfrecords/retinanet/yolov7_tmp4/train*',
                    'Where to find tfrecords for inference.')
flags.DEFINE_string('model_path',
                    None,
                    'Where to load model for inference.') # download from gs://model-chkpts/shaili/tfvision_test_yolov7x_eval_full/tf29
flags.DEFINE_string('export_dir',
                    None,
                    'The export directory where images with BBOX can be saved.')
flags.DEFINE_string(
    'input_image_size', '640,640',
    'The comma-separated string of two integers representing the height,width '
    'of the input to the model.')
category_index = {
    1: {'id': 1, 'name': 'black'},
    2: {'id': 2, 'name': 'non-black'},
    3: {'id': 3, 'name': 'none'}
    }


def build_inputs_for_object_detection(image, input_image_size):
  """Builds Object Detection model inputs for serving."""
  image, _ = resize_and_crop_image(
      image,
      input_image_size,
      padded_size=input_image_size,
      aug_scale_min=1.0,
      aug_scale_max=1.0,
  )
  return image


def save_image(tfrecords, input_image_size, model_fn, decoder, export_dir):
  for i, serialized_example in enumerate(tfrecords):
    decoded_tensors = decoder.decode(serialized_example)
    image = build_inputs_for_object_detection(
        decoded_tensors['image'], input_image_size
    )
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.uint8)
    image_np = image[0].numpy()
    result = model_fn(image)
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        result['detection_boxes'][0].numpy(),
        result['detection_classes'][0].numpy().astype(int),
        result['detection_scores'][0].numpy(),
        category_index=category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=MIN_SCORE_THRESH,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4,
    )
    save_path = os.path.join(export_dir, str(i) + '.PNG')
    cv2.imwrite(save_path, image_np)


def main(_):
  input_image_size = [int(x) for x in FLAGS.input_image_size.split(',')]
  decoder = TfExampleDecoder()

  # load data
  tfrecords = (
      tf.data.TFRecordDataset(tf.io.gfile.glob(FLAGS.tfrecords_path))
      .shuffle(buffer_size=BUFFER_SIZE)
      .take(FLAGS.num_of_examples)
  )

  # load model
  imported = tf.saved_model.load(FLAGS.model_path)
  model_fn = imported.signatures['serving_default']

  # save predicted images
  save_image(tfrecords, input_image_size, model_fn, decoder, FLAGS.export_dir)

if __name__ == '__main__':
  app.run(main)
