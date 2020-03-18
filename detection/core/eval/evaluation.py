import tensorflow as tf
import json
import numpy as np
from pycocotools.cocoeval import COCOeval
from time import time
import pathlib

def get_detections_single(detection, img_id, meta):
    detection = tf.gather_nd(detection, tf.where(tf.reduce_mean(detection, axis=1)))
    boxes = detection[...,:4].numpy()
    labels = detection[...,4].numpy()
    probs = detection[...,5].numpy()
    boxes = boxes[...,[1, 0, 3, 2]]/meta[-2].numpy()
    boxes[..., 2] = boxes[..., 2] - boxes[..., 0] 
    boxes[..., 3] = boxes[..., 3] - boxes[..., 1] 
    json_list = []
    for box, label, score in zip(boxes, labels, probs):
        json_list.append({"image_id": int(img_id.numpy()[0]), "category_id": int(label), "bbox": box.tolist(), "score": float(score)})
    return json_list

def get_detection_batch(detections_batch, img_id_batch, meta_batch):
    json_list = []
    for detection, img_id, meta in zip(detections_batch, img_id_batch, meta_batch):
        json_list.extend(get_detections_single(detection, img_id, meta))
    return json_list

def get_detections(detections_list, img_ids, metas):
    json_list = []
    for detections_batch, img_id_batch, meta_batch in zip(detections_list, img_ids, metas):
        json_list.extend(get_detection_batch(detections_batch, img_id_batch, meta_batch))
    return json_list

def run_eval(detections_list, img_ids, metas, val_dataset, filename=None):
    if not filename:
        filename = "eval_pred_{}.json".format(int(time()*10000))
    json_list = get_detections(detections_list, img_ids, metas)
    with open(filename, 'w') as outfile:
        outfile.write(json.dumps(json_list))
    cocoDt = val_dataset.coco.loadRes(filename)
    cocoEval = COCOeval(val_dataset.coco, cocoDt , 'bbox')
    cocoEval.params.imgIds = tf.reshape(tf.concat(img_ids, axis=0), [-1]).numpy().tolist()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    pathlib.Path(filename).unlink()