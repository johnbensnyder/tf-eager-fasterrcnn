import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import horovod.tensorflow as hvd
from tqdm import tqdm
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.ops import clip_ops

from detection.datasets import coco, data_generator

from detection.models.detectors import faster_rcnn
from detection.utils import schedulers

hvd.init()
# mpirun -np 4 -H localhost:4 --bind-to none --allow-run-as-root python rpn_hvd.py
#tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


#img_mean = (123.675, 116.28, 103.53)
img_mean = (123.68, 116.779, 103.939)
# img_std = (58.395, 57.12, 57.375)
#img_mean = (127.5, 127.5, 127.5)
img_std = (1., 1., 1.)
#img_std = (127.5, 127.5, 127.5)
images = 118000
batch_size = 8
learning_rate = batch_size*hvd.size()*5e-4
warmup_rate = learning_rate/10
warmup_steps = 2000
loss_weights = [1, 1, 1, 1]
steps_per_epoch = images//(batch_size*hvd.size())
train_dataset = coco.CocoDataSet('/workspace/shared_workspace/data/coco', 'train',
                                 flip_ratio=0.5,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1216))

train_generator = data_generator.DataGenerator(train_dataset, shuffle=True)
train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))

train_tf_dataset = train_tf_dataset.prefetch(256) #.shuffle(100).shard(hvd.size(), hvd.rank())
train_tf_dataset = train_tf_dataset.padded_batch(
                            batch_size,
                            padded_shapes=(
                            tf.TensorShape([None, None, 3]), # image padded to largest in batch
                            tf.TensorShape([11]),            # image meta - no padding
                            tf.TensorShape([None, 4]),       # bounding boxes, padded to longest
                            tf.TensorShape([None])           # labels, padded to longest
                            ),
                            padding_values=(0.0, 0.0, 0.0, -1))

# flip channel order
def flip_channels(img, img_meta, bbox, label):
    img = tf.reverse(img, axis=[-1])
    return img, img_meta, bbox, label

train_tf_dataset = train_tf_dataset.filter(lambda w, x, y, z: tf.equal(tf.shape(w)[0], batch_size))
train_tf_dataset = train_tf_dataset.map(flip_channels, num_parallel_calls=16)
train_tf_dataset = iter(train_tf_dataset.repeat())

model = faster_rcnn.FasterRCNN(
    num_classes=len(train_dataset.get_categories()), batch_size=batch_size)

for i in train_tf_dataset:
    _ = model(i)
    break

#model.layers[0].load_weights('resnet_101_backbone.h5')
'''for layer in model.layers[0].layers[0].layers[:142]:
    layer.trainable=False'''
#model.layers[0].trainable=False
#model.layers[4].trainable=False
#model.layers[0].load_weights('resnet_101_backbone.h5')
model.layers[0].trainable=False
'''for layer in model.layers[0].layers[0].layers[80:]:
    if type(layer)!=BatchNormalization:
        layer.trainable=True'''

scheduler = schedulers.WarmupExponentialDecay(warmup_rate, learning_rate, warmup_steps,
                                              steps_per_epoch*12, learning_rate*1e-1)
#scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay([steps_per_epoch*5],
#                                                                 [1e-3, 1e-4])
#optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True, clipnorm=5.0)
#optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)
optimizer = tfa.optimizers.SGDW(1e-4, scheduler, momentum=0.9, nesterov=True)
#optimizer = tfa.optimizers.AdamW(1e-4, scheduler)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

@tf.function(experimental_relax_shapes=True)
def train_step(inputs):
    batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
    with tf.GradientTape() as tape:
        rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
            model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
        loss_value = (loss_weights[0] * rpn_class_loss \
                      + loss_weights[1] * rpn_bbox_loss \
                      + loss_weights[2] * rcnn_class_loss \
                      + loss_weights[3] * rcnn_bbox_loss)
        scaled_loss = optimizer.get_scaled_loss(loss_value)
    tape = hvd.DistributedGradientTape(tape)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    #grads = tape.gradient(loss_value, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    grads = [tf.clip_by_norm(g, 2.0) for g in grads]
    grads = [grad if grad is not None \
             else tf.zeros_like(var) \
             for var, grad in zip(model.trainable_variables, grads)]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss

def train_epoch(epochs=1, filename=None):
    for epoch in range(epochs):
        rpn_class_loss_history = []
        rpn_bbox_loss_history = []
        rcnn_class_loss_history = []
        rcnn_bbox_loss_history = [] 
        if hvd.rank()==0:
            progressbar = tqdm(range(steps_per_epoch))
            for batch in progressbar:
                inputs = next(train_tf_dataset)
                rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = train_step(inputs)
                rpn_class_loss_history.append(rpn_class_loss)
                rpn_bbox_loss_history.append(rpn_bbox_loss)
                rcnn_class_loss_history.append(rcnn_class_loss)
                rcnn_bbox_loss_history.append(rcnn_bbox_loss)
                progressbar.set_description("rpnc: {0:.5f} rpnb {1:.5f} rcnc {2:.5f} rcnb {3:.5f} lr {4:.5f}". \
                                            format(np.array(rpn_class_loss_history[-200:]).mean(),
                                            np.array(rpn_bbox_loss_history[-200:]).mean(),
                                            np.array(rcnn_class_loss_history[-200:]).mean(),
                                            np.array(rcnn_bbox_loss_history[-200:]).mean(),
                                            scheduler(optimizer.iterations)))

        else:
            for batch in range(steps_per_epoch):
                inputs = next(train_tf_dataset)
                rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = train_step(inputs)
    if hvd.rank()==0 and filename:
        model.save_weights(filename)
    
train_epoch(epochs=2, filename = 'rcnn_keras_resnet_50_stage_1.h5')

    
for layer in model.layers[0].layers[0].layers[142:]:
    if type(layer)!=BatchNormalization:
        layer.trainable=True

train_epoch(epochs=4, filename = 'rcnn_keras_resnet_50_stage_2.h5')

for layer in model.layers[0].layers[0].layers[80:]:
    if type(layer)!=BatchNormalization:
        layer.trainable=True
        
train_epoch(epochs=6, filename = 'rcnn_keras_resnet_50_stage_3.h5')