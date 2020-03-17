from pathlib import Path
import tensorflow as tf

class TFRecordCoco(object):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            'label/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
            'label/format': tf.io.FixedLenFeature((), tf.string, default_value='png'),
            'image/height': tf.io.FixedLenFeature((), tf.int64),
            'image/width': tf.io.FixedLenFeature((), tf.int64),
            'image/object/bbox/xmin' : tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax' : tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin' : tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax' : tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string)
          }
        self.label_lookup, self.cat_lookup = self._make_lookup()
    
    def make_dataset(self, subset='train', threads=12, shards=1, index=0):
        files = [i.as_posix() for i in self.data_dir.glob('*{}*'.format(subset))]
        shard_length = len(files)//shards
        files = files[index*shard_length:(index+1)*shard_length]
        tdf = tf.data.TFRecordDataset(files)
        tdf = tdf.map(self.parse, num_parallel_calls=threads)
        tdf = tdf.filter(self.record_filter)
        return tdf
    
    def _make_lookup(self):
        label_list = {0: 'unlabeled', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 
                      5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
                      11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 
                      15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 
                      22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 
                      28: 'umbrella', 29: 'shoe', 30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 
                      34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 
                      39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 
                      43: 'tennis racket', 44: 'bottle', 45: 'plate', 46: 'wine glass', 47: 'cup', 
                      48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 
                      54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 
                      59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 
                      65: 'bed', 66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet', 
                      71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 
                      77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 
                      82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 
                      88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush', 91: 'hair brush', 92: 'banner', 
                      93: 'blanket', 94: 'branch', 95: 'bridge', 96: 'building-other', 97: 'bush', 
                      98: 'cabinet', 99: 'cage', 100: 'cardboard', 101: 'carpet', 102: 'ceiling-other', 
                      103: 'ceiling-tile', 104: 'cloth', 105: 'clothes', 106: 'clouds', 107: 'counter', 
                      108: 'cupboard', 109: 'curtain', 110: 'desk-stuff', 111: 'dirt', 112: 'door-stuff', 
                      113: 'fence', 114: 'floor-marble', 115: 'floor-other', 116: 'floor-stone', 
                      117: 'floor-tile', 118: 'floor-wood', 119: 'flower', 120: 'fog', 121: 'food-other', 
                      122: 'fruit', 123: 'furniture-other', 124: 'grass', 125: 'gravel', 126: 'ground-other', 
                      127: 'hill', 128: 'house', 129: 'leaves', 130: 'light', 131: 'mat', 132: 'metal', 
                      133: 'mirror-stuff', 134: 'moss', 135: 'mountain', 136: 'mud', 137: 'napkin', 
                      138: 'net', 139: 'paper', 140: 'pavement', 141: 'pillow', 142: 'plant-other', 
                      143: 'plastic', 144: 'platform', 145: 'playingfield', 146: 'railing', 147: 'railroad', 
                      148: 'river', 149: 'road', 150: 'rock', 151: 'roof', 152: 'rug', 153: 'salad', 
                      154: 'sand', 155: 'sea', 156: 'shelf', 157: 'sky-other', 158: 'skyscraper', 
                      159: 'snow', 160: 'solid-other', 161: 'stairs', 162: 'stone', 163: 'straw', 
                      164: 'structural-other', 165: 'table', 166: 'tent', 167: 'textile-other', 168: 'towel', 
                      169: 'tree', 170: 'vegetable', 171: 'wall-brick', 172: 'wall-concrete', 173: 'wall-other', 
                      174: 'wall-panel', 175: 'wall-stone', 176: 'wall-tile', 177: 'wall-wood', 178: 'water-other', 
                      179: 'waterdrops', 180: 'window-blind', 181: 'window-other', 182: 'wood'}
        label_list = {j:i for i,j in label_list.items()}
        label_lookup = tf.lookup.StaticHashTable(
               tf.lookup.KeyValueTensorInitializer(tf.constant(list(label_list.keys())), list(label_list.values())), 0)
        label_to_cat = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 
                        17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 
                        33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 
                        47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 
                        60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67,
                         77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
        cat_lookup = tf.lookup.StaticHashTable(
               tf.lookup.KeyValueTensorInitializer(tf.constant(list(label_to_cat.keys())), list(label_to_cat.values())), 0)
        return label_lookup, cat_lookup
    
    def scale_bbox(self, x_min, x_max, y_min, y_max, image_shape):
        x_min = x_min*image_shape[1]
        x_max = x_max*image_shape[1]
        y_min = y_min*image_shape[0]
        y_max = y_max*image_shape[0]
        return x_min.values, x_max.values, y_min.values, y_max.values
    
    def flip_image(self, image, bbox, original_image_shape, flip):
        if flip:
            new_coords = original_image_shape[1] - bbox[:,0] - bbox[:,2]
            bbox = tf.stack([new_coords, bbox[:,1], bbox[:,2], bbox[:,3]], axis=1)
            image = tf.image.flip_left_right(image)
        return image, bbox

    def pad_image(self, image, pad_shape):
        y_shape = tf.shape(image)[1]
        x_shape = tf.shape(image)[0]
        return tf.pad(image, [[0, pad_shape-x_shape], [0, pad_shape-y_shape], [0,0]])

    def img_meta(self, original_image_shape, scale_factor, flip, pad_shape):
            return tf.concat([original_image_shape, 
                              original_image_shape[:2]*scale_factor, 
                              tf.constant([3.]),
                              [tf.cast(pad_shape, tf.float32)],
                              [tf.cast(pad_shape, tf.float32)],
                              tf.constant([3.]),
                              [scale_factor],
                              [flip]], axis=0)

        
    def record_filter(self, image, meta, bbox, labels):
        '''
        filter records with no boxes
        '''
        return len(labels)>0
    
    def parse(self, record):
        record = tf.io.parse_single_example(record, self.keys_to_features)
        image = record['image/encoded']
        image = tf.image.decode_jpeg(image)
        original_image_shape = tf.cast(tf.shape(image), tf.float32)
        if tf.shape(image)[-1]==1:
            image = tf.image.grayscale_to_rgb(image)
        img_mean = (123.68, 116.779, 103.939)
        # img_std = (58.395, 57.12, 57.375)
        img_std = (1., 1., 1.)
        image = (tf.cast(image, tf.float32) - img_mean)/img_std
        image = tf.reverse(image, axis=[-1])
        #image = tf.image.per_image_standardization(image)
        x_min, x_max, y_min, y_max = self.scale_bbox(record['image/object/bbox/xmin'],
                                                record['image/object/bbox/xmax'], 
                                                record['image/object/bbox/ymin'], 
                                                record['image/object/bbox/ymax'], original_image_shape)
        labels = self.cat_lookup.lookup(self.label_lookup.lookup(record['image/object/class/text'])).values
        bbox = tf.stack([y_min, x_min, y_max, x_max], axis=-1)
        #flip = tf.random.uniform(())<.5
        flip = 0
        #image, bbox = flip_image(image, bbox, original_image_shape, flip)
        image = tf.image.resize(image, (1216, 1216), 
                                preserve_aspect_ratio=True)
        image = self.pad_image(image, 1216)
        scale_factor = 1216/tf.reduce_max(original_image_shape[:2])
        #scale_factor = 1
        bbox = bbox * scale_factor
        meta = self.img_meta(original_image_shape, scale_factor, flip, 1216)
        return image, meta, bbox, labels