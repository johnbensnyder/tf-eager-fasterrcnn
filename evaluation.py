import os
import tensorflow as tf
import numpy as np
import visualize
import json

from detection.datasets import coco, data_generator
from detection.datasets.utils import get_original_image
from detection.models.detectors import faster_rcnn
from pycocotools.cocoeval import COCOeval

IMG_MEAN = (123.68, 116.779, 103.939) #(123.675, 116.28, 103.53)
IMG_STD = (1., 1., 1.)
IMG_SCALE=(800, 1216)
NUM_CLASSES=81
DATASET_ROOT='/data/COCO'
WEIGHTS_PATH='weights/rcnn101_sgdw_training_epoch_1.h5' #rcnn_101_training_epoch_3.h5'

val_dataset = coco.CocoDataSet(DATASET_ROOT, 'val',
                                 flip_ratio=0.0,
                                 pad_mode='fixed',
                                 mean=IMG_MEAN,
                                 std=IMG_STD,
                                 scale=IMG_SCALE)
val_generator = data_generator.DataGenerator(val_dataset, shuffle=False)
val_tf_dataset = tf.data.Dataset.from_generator(
    val_generator, (tf.float32, tf.float32, tf.float32, tf.int32)).prefetch(100)
val_tf_dataset = iter(val_tf_dataset.repeat())

model = faster_rcnn.FasterRCNN(num_classes=len(val_dataset.get_categories()))

img, img_meta, bboxes, labels = val_dataset[0]
batch_imgs = tf.Variable(np.expand_dims(img, 0))
batch_metas = tf.Variable(np.expand_dims(img_meta, 0))

model.layers[0].trainable=False
_ = model((batch_imgs, batch_metas), training=False)

# print(model.layers)
# print('before', model.layers[4].get_weights())
#model.layers[0].trainable=False
model.load_weights(WEIGHTS_PATH, by_name=True) # , skip_mismatch=True)
# print('after', model.layers[4].get_weights())


def eval_step(inputs):
    img, img_meta, _, _ = inputs
    rcnn_feature_maps, img_metas, proposals = model.simple_test_rpn_tf(img, img_meta)
    res = model.simple_test_bboxes_tf(rcnn_feature_maps, img_metas, proposals)
    return res


dataset_results = []
imgIds = []
for idx in range(len(val_dataset)):
    #print('img id:', val_dataset.img_ids[idx])
    if idx > 0 and idx % 100 == 0:
        print(idx)
        #break
    inputs = next(val_tf_dataset)
    res = eval_step(inputs)
    image_id = val_dataset.img_ids[idx]
    imgIds.append(image_id)
 
    for pos in range(res['class_ids'].shape[0]):
        results = dict()
        results['score'] = float(res['scores'][pos])
        results['category_id'] = int(res['class_ids'][pos])
        y1, x1, y2, x2 = [float(num) for num in list(res['rois'][pos])]
        results['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        results['image_id'] = image_id
        dataset_results.append(results)

with open('coco_val2017_detection_result.json', 'w') as f:
    f.write(json.dumps(dataset_results))

coco_dt = val_dataset.coco.loadRes('coco_val2017_detection_result.json')
cocoEval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

