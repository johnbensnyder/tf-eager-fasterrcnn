import tensorflow as tf

from detection.core.bbox import geometry, transforms
from detection.utils.misc import trim_zeros

class AnchorTarget(object):
    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 neg_multiplier=1,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 batch_size=1):
        '''Compute regression and classification targets for anchors.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RPN.
            target_stds: [4]. Bounding box refinement standard deviation for RPN.
            num_rpn_deltas: int. Maximal number of Anchors per image to feed to rpn heads.
            neg_multiplier: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.neg_multiplier = neg_multiplier
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.batch_size = batch_size
   
    def compute_argmax(self, ious):
        '''
        Need to compute max overlap with gt for each anchor
        BUT also prioritize gt max values, so overright
        argmax with new gt label if that gt needs it
        '''
        anchor_iou_argmax = tf.argmax(ious, axis=2)
        gt_iou_argmax = tf.argmax(ious, axis=1)
        #default_values = tf.tile(tf.expand_dims(tf.gather_nd(anchor_iou_argmax, 
        #                   tf.concat([tf.expand_dims(tf.repeat(0, tf.shape(ious)[0]), axis=1),
        #                        tf.expand_dims(tf.range(tf.shape(ious)[0]), axis=1)], axis=1)), axis=1), [1, tf.shape(ious)[2]])
        fill_values = tf.where(gt_iou_argmax!=0)[..., 1]
        positions = tf.transpose(tf.stack([tf.where(gt_iou_argmax!=0)[..., 0], 
                                       tf.gather_nd(gt_iou_argmax, tf.where(gt_iou_argmax!=0)),
                                       fill_values]))
        positions = tf.boolean_mask(positions, tf.gather_nd(ious, positions)<self.pos_iou_thr)
        fill_values = positions[...,2]
        positions = positions[...,:2]
        anchor_iou_argmax = tf.tensor_scatter_nd_update(anchor_iou_argmax, positions, fill_values)
        return anchor_iou_argmax
    
    def fill_missing_gts(self, target_matches, ious):
        '''
        For gts that didn't get a box assignment,
        assign the highest overlap
        '''
        '''gt_iou_argmax = tf.argmax(ious, axis=1, output_type=tf.int32)
        default_values = tf.tile(tf.expand_dims(tf.gather_nd(target_matches, 
                           tf.concat([tf.expand_dims(tf.repeat(0, tf.shape(ious)[0]), axis=1),
                                tf.expand_dims(tf.range(tf.shape(ious)[0]), axis=1)], axis=1)), axis=1), [1, tf.shape(ious)[2]])
        fill_values = tf.reshape(tf.where(gt_iou_argmax!=0, tf.ones(tf.shape(gt_iou_argmax), dtype=tf.int32) , default_values), [-1])
        position_values = tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(ious)[0]), axis=1), [1, tf.shape(ious)[2]]), [-1]), axis=1)
        gt_positions = tf.expand_dims(tf.reshape(gt_iou_argmax, [-1]), axis=1)
        gt_positions = tf.concat([position_values, gt_positions], axis=1)
        target_matches = tf.tensor_scatter_nd_update(target_matches, gt_positions, fill_values)'''
        gt_iou_argmax = tf.argmax(ious, axis=1)
        positions = tf.where(gt_iou_argmax!=0)
        gt_positions = tf.gather_nd(gt_iou_argmax, positions)
        gt_positions = tf.transpose(tf.stack([positions[...,0], gt_positions]))
        fill_values = tf.ones(tf.shape(gt_positions)[0], dtype=tf.int32)
        target_matches = tf.tensor_scatter_nd_update(target_matches, gt_positions, fill_values)
        return target_matches
    
    def subset_targets(self, target_matches):
        pos_ids = tf.where(tf.equal(target_matches, 1))
        pos_ids = tf.gather(pos_ids, 
                 tf.concat([tf.sort(tf.random.shuffle(tf.reshape(tf.where(pos_ids[...,0]==i), 
                                        [-1]))[:self.num_rpn_deltas]) for i in range(self.batch_size)], axis=0))
        neg_ids = tf.where(tf.equal(target_matches, -1))
        neg_ids = tf.gather(neg_ids, 
                  tf.concat([tf.random.shuffle(tf.reshape(tf.where(neg_ids[...,0]==i), [-1])) \
                    [:tf.math.minimum(tf.shape(tf.where(pos_ids[...,0]==i))[0]*self.neg_multiplier,
                                                                  self.num_rpn_deltas)] \
                             for i in range(self.batch_size)], axis=0))
        '''neg_ids = tf.gather(neg_ids, 
                  tf.concat([tf.random.shuffle(tf.reshape(tf.where(neg_ids[...,0]==i), [-1])) \
                                                [:self.num_rpn_deltas] \
                             for i in range(self.batch_size)], axis=0))'''
        target_matches = tf.zeros(tf.shape(target_matches), dtype=tf.int32)
        target_matches = tf.tensor_scatter_nd_update(target_matches, pos_ids, tf.ones(tf.shape(pos_ids)[0], dtype=tf.int32))
        target_matches = tf.tensor_scatter_nd_update(target_matches, neg_ids, -tf.ones(tf.shape(neg_ids)[0], dtype=tf.int32))
        return pos_ids, neg_ids, target_matches
    
    def batch_iou(self, anchors, bboxes):
        # replace this line, reshape anchors outside
        anchors = tf.tile(tf.expand_dims(anchors, axis=0), [tf.shape(bboxes)[0], 1, 1])
        y1_min, x1_min, y1_max, x1_max = tf.split(
            value=anchors, num_or_size_splits=4, axis=2)
        y2_min, x2_min, y2_max, x2_max = tf.split(
            value=bboxes, num_or_size_splits=4, axis=2)
        intersection_xmin = tf.maximum(x1_min, tf.transpose(x2_min, [0, 2, 1]))
        intersection_xmax = tf.minimum(x1_max, tf.transpose(x2_max, [0, 2, 1]))
        intersection_ymin = tf.maximum(y1_min, tf.transpose(y2_min, [0, 2, 1]))
        intersection_ymax = tf.minimum(y1_max, tf.transpose(y2_max, [0, 2, 1]))
        intersection_area = tf.maximum(
            (intersection_xmax - intersection_xmin), 0) * tf.maximum(
                (intersection_ymax - intersection_ymin), 0)
        area1 = (y1_max - y1_min) * (x1_max - x1_min)
        area2 = (y2_max - y2_min) * (x2_max - x2_min)
        union_area = area1 + tf.transpose(area2, [0, 2, 1]) - intersection_area + 1e-8
        iou = intersection_area / union_area
        padding_mask = tf.logical_and(tf.less(intersection_xmax, 0), tf.less(intersection_ymax, 0))
        iou = tf.where(padding_mask, -tf.ones_like(iou), iou)
        return iou
    
    #@tf.function
    def get_counts(self, anchor_idx, batch_size):
        per_image_count = tf.unique_with_counts(anchor_idx[...,0], out_idx=tf.int64)[2]
        count = tf.cast(tf.math.cumsum(tf.ones(tf.shape(anchor_idx)[0])), tf.int64)
        counts = tf.concat([tf.constant([0], dtype=tf.int64), count], axis=0)
        target_deltas_counts = tf.concat([counts[:per_image_count[i]] for i in range(batch_size)], axis=-1)
        return target_deltas_counts
    
    @tf.function(experimental_relax_shapes=True)
    def pad_target_deltas(self, target_deltas, pos_ids):
        reshaped_images = tf.TensorArray(tf.float32, size=self.batch_size)
        max_deltas = tf.reduce_max(tf.unique_with_counts(pos_ids[...,0])[2])
        for i in range(self.batch_size):
            image_gt = tf.gather_nd(target_deltas, tf.where(pos_ids[...,0]==i))
            image_gt = tf.pad(image_gt, [[0,max_deltas-tf.shape(image_gt)[0]], [0,0]])
            reshaped_images = reshaped_images.write(i, image_gt)
        return reshaped_images.stack()
    
    def build_targets(self, anchors, valid_flags, bboxes, labels):
        anchor_size = tf.shape(anchors)[0]
        ious = self.batch_iou(anchors, bboxes)
        anchor_iou_argmax = self.compute_argmax(ious)
        anchor_iou_max = tf.reduce_max(ious, axis=2)
        target_matches = tf.zeros((self.batch_size, anchor_size), dtype=tf.int32)
        # get negative values
        target_matches = tf.where(anchor_iou_max < self.neg_iou_thr, 
                                -tf.ones((self.batch_size, anchor_size), dtype=tf.int32), target_matches)
        # filter invalid flags
        target_matches = tf.where(tf.equal(valid_flags, 1),
                                         target_matches, tf.zeros((self.batch_size, anchor_size), dtype=tf.int32))
        # get pos anchors
        target_matches = tf.where(anchor_iou_max >= self.pos_iou_thr, 
                                  tf.ones((self.batch_size, anchor_size), dtype=tf.int32), target_matches)
        # fill in missing gt values
        target_matches = self.fill_missing_gts(target_matches, ious)
        pos_ids, neg_ids, target_matches = self.subset_targets(target_matches)
        a = tf.gather_nd(tf.tile(tf.expand_dims(anchors, axis=0), [self.batch_size, 1, 1]), pos_ids)
        anchor_idx = tf.gather_nd(anchor_iou_argmax, pos_ids)
        anchor_idx = tf.transpose(tf.stack([pos_ids[...,0], anchor_idx]))
        gt = tf.gather_nd(bboxes, anchor_idx)
        target_deltas = transforms.bbox2delta(a, gt, self.target_means, self.target_stds)
        # reshape to batch
        target_deltas = self.pad_target_deltas(target_deltas, pos_ids)
        return target_matches, target_deltas
    