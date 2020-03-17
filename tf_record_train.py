import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import horovod.tensorflow as hvd
from tqdm import tqdm
from tensorflow.python.keras.layers.normalization import BatchNormalization

from detection.datasets import coco, data_generator, tf_record

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
batch_size = 1
loss_weights = [1, 1, 1, 1]
steps_per_epoch = images//(batch_size*hvd.size())

tf_record_creator = tf_record.TFRecordCoco('/workspace/shared_workspace/data/coco/tfrecord')

train_tf_dataset = iter(tf_record_creator.make_dataset().prefetch(256).shuffle(128).batch(batch_size).repeat())

model = faster_rcnn.FasterRCNN(
    num_classes=81, batch_size=batch_size)

for i in train_tf_dataset:
    _ = model(i)
    break
    
model.layers[0].trainable=False
for layer in model.layers[0].layers[0].layers[80:]:
    if type(layer)!=BatchNormalization:
        layer.trainable=True
        
scheduler = schedulers.WarmupExponentialDecay(1e-3, 1e-2, steps_per_epoch,
                                              1e-4, steps_per_epoch*12)

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
        #scaled_loss = optimizer.get_scaled_loss(loss_value)
    #tape = hvd.DistributedGradientTape(tape)
    #scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = tape.gradient(loss_value, model.trainable_variables)
    #grads = optimizer.get_unscaled_gradients(scaled_grads)
    grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, grads)]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss

total_epochs = 1

epochs = 12
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
            progressbar.set_description("rpnc: {0:.5f} rpnb {1:.5f} rcnc {2:.5f} rcnb {3:.5f}". \
                                        format(np.array(rpn_class_loss_history[-200:]).mean(),
                                        np.array(rpn_bbox_loss_history[-200:]).mean(),
                                        np.array(rcnn_class_loss_history[-200:]).mean(),
                                        np.array(rcnn_bbox_loss_history[-200:]).mean()))
                                        
        model.save_weights("rcnn101_sgdw_training_epoch_{}.h5".format(total_epochs))
        total_epochs+=1
    else:
        for batch in range(steps_per_epoch):
            inputs = next(train_tf_dataset)
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = train_step(inputs)
            
for layer in model.layers[0].layers[0].layers[142:]:
    if type(layer)!=BatchNormalization:
        layer.trainable=True


