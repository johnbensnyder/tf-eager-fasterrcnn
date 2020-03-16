import tensorflow as tf

class ResNet(tf.keras.Model):
    def __init__(self, depth=50, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        self.model = tf.keras.Model(inputs = self.base_model.input, 
                                    outputs = [self.base_model.layers[i].output for i in [38, 80, 142, 174]])
    
    @tf.function
    def call(self, inputs, training=True):
        c2, c3, c4, c5 = self.model(inputs, training)
        return (c2, c3, c4, c5)
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        batch, H, W, C = shape
        
        C2_shape = tf.TensorShape([batch, H //  4, W //  4, self.out_channel[0]])
        C3_shape = tf.TensorShape([batch, H //  8, W //  8, self.out_channel[1]])
        C4_shape = tf.TensorShape([batch, H // 16, W // 16, self.out_channel[2]])
        C5_shape = tf.TensorShape([batch, H // 32, W // 32, self.out_channel[3]])
        
        return (C2_shape, C3_shape, C4_shape, C5_shape)