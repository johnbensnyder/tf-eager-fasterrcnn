import os
import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
from tqdm import tqdm

from detection.datasets import coco, data_generator

from detection.models.detectors import faster_rcnn

hvd.init()
# mpirun -np 4 -H localhost:4 --bind-to none --allow-run-as-root python rpn_hvd.py
#tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

train_dataset = coco.CocoDataSet('/workspace/shared_workspace/data/coco', 'train',
                                 flip_ratio=0.5,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1216))

train_generator = data_generator.DataGenerator(train_dataset, shuffle=True)
train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
batch_size = 4
train_tf_dataset = train_tf_dataset.prefetch(128) #.shuffle(100).shard(hvd.size(), hvd.rank())
train_tf_dataset = train_tf_dataset.padded_batch(
                            batch_size,
                            padded_shapes=(
                            tf.TensorShape([None, None, 3]), # image padded to largest in batch
                            tf.TensorShape([11]),            # image meta - no padding
                            tf.TensorShape([100, 4]),       # bounding boxes, padded to longest
                            tf.TensorShape([100])           # labels, padded to longest
                            ),
                            padding_values=(0.0, 0.0, 0.0, -1))

model = faster_rcnn.FasterRCNN(
    num_classes=len(train_dataset.get_categories()))

for i in train_tf_dataset:
    _ = model(i)
    break
    
model.layers[0].load_weights('resnet_101_backbone.h5')
model.layers[0].trainable=False
#model.load_weights('rpn_training.h5')

#optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True, clipnorm=5.0)
optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

@tf.function(experimental_relax_shapes=True)
def train_step(inputs):
    batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
    with tf.GradientTape() as tape:
        rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
            model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
        loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss
        scaled_loss = optimizer.get_scaled_loss(loss_value)
    tape = hvd.DistributedGradientTape(tape)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    #grads = tape.gradient(loss_value, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, grads)]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss

epochs = 4
for epoch in range(epochs):
    rpn_class_loss_history = []
    rpn_bbox_loss_history = []
    rcnn_class_loss_history = []
    rcnn_bbox_loss_history = []
    if hvd.rank()==0:
        progressbar = tqdm(enumerate(train_tf_dataset))
        for (batch, inputs) in progressbar:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = train_step(inputs)
            rpn_class_loss_history.append(rpn_class_loss)
            rpn_bbox_loss_history.append(rpn_bbox_loss)
            rcnn_class_loss_history.append(rcnn_class_loss)
            rcnn_bbox_loss_history.append(rcnn_bbox_loss)
            progressbar.set_description("rpnc: {0:.5f} rpnb {1:.5f} rcnc {2:.5f} rcnb {3:.5f}". \
                                        format(np.array(rpn_class_loss_history[-100:]).mean(),
                                        np.array(rpn_bbox_loss_history[-100:]).mean(),
                                        np.array(rcnn_class_loss_history[-100:]).mean(),
                                        np.array(rcnn_bbox_loss_history[-100:]).mean()))
                                        
        model.save_weights("rpn_training_epoch_{}.h5".format(epoch+1))
    else:
        for (batch, inputs) in enumerate(train_tf_dataset):
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = train_step(inputs)

