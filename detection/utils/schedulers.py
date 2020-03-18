import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler that linearly scales up during the first epoch
    then decays exponentially
    """
    def __init__(self, initial_rate, scaled_rate, steps_at_scale, decay_steps, decay_rate, name=None):
        super(WarmupExponentialDecay, self).__init__()
        self.initial_rate = initial_rate
        self.scaled_rate = scaled_rate
        self.steps_at_scale = steps_at_scale
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.name = name
    
    @tf.function
    def __call__(self, step, dtype=tf.float32):
        initial_learning_rate = math_ops.cast(self.initial_rate, dtype)
        decay_steps = math_ops.cast(self.decay_steps, dtype)
        decay_rate = math_ops.cast(self.decay_rate, dtype)
        steps_at_scale = math_ops.cast(self.steps_at_scale, dtype)
        scaled_rate = math_ops.cast(self.scaled_rate, dtype)
        global_step_recomp = math_ops.cast(step, dtype)
        
        if step<=self.steps_at_scale:
            return self.compute_warmup(global_step_recomp, initial_learning_rate, scaled_rate, steps_at_scale)
        else:
            return self.compute_decay(global_step_recomp, scaled_rate, steps_at_scale, decay_rate, decay_steps)
    
    def compute_warmup(self, step, initial_learning_rate, scaled_rate, steps_at_scale):
        return ((scaled_rate*step)+(initial_learning_rate*(steps_at_scale-step)))/steps_at_scale
    
    def compute_decay(self, step, scaled_rate, steps_at_scale, decay_rate, decay_steps):
        return scaled_rate*decay_rate**((step-steps_at_scale)/decay_steps)
    
    def get_config(self):
        return {"initial_rate": self.initial_rate, "scaled_rate": self.scaled_rate, 
                "steps_at_scale": self.steps_at_scale, "decay_rate": self.decay_rate, 
                "decay": self.decay_steps, "name": self.name}
    
class WarmupPiecewiseConstantDecay(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
    
    def __init__(self, initial_rate, steps_at_scale, boundaries, values, dtype=tf.float32, name=None):
        super().__init__(boundaries, values)
        self.dtype = dtype
        self.initial_learning_rate = math_ops.cast(initial_rate, dtype)
        self.steps_at_scale = math_ops.cast(steps_at_scale, dtype)
        self.scaled_rate = math_ops.cast(values[0], dtype)
        self.boundaries = boundaries
        self.values = values
        self.name = name
        
    @tf.function
    def __call__(self, step):
        global_step_recomp = math_ops.cast(step, self.dtype)
        if global_step_recomp>=self.steps_at_scale:
            return super().__call__(step)
        return self.compute_warmup(global_step_recomp)
        
    def compute_warmup(self, step):
        return ((self.scaled_rate*step)+(self.initial_learning_rate*(self.steps_at_scale-step)))/self.steps_at_scale
    
    def get_config(self):
        return {"initial_rate": self.initial_rate, "scaled_rate": self.scaled_rate, 
                "steps_at_scale": self.steps_at_scale, "boundaries": self.boundaries, 
                "values": self.values, "name": self.name}