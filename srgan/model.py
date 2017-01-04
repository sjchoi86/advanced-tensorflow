import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Model:    
    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]
    def _get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        return '%s_L%03d' % (self.name, layer+1)
    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])
    def _glorot_initializer(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.
        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)
    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.
        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)
    def get_num_layers(self):
        return len(self.outputs)
    def add_batch_norm(self, scale=False):
        """Adds a batch normalization layer to this model.
        See ArXiv 1502.03167v3 for details."""
        # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
        with tf.variable_scope(self._get_layer_str()):
            out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale)
        self.outputs.append(out)
        return self
    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""
        with tf.variable_scope(self._get_layer_str()):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, -1])
        self.outputs.append(out)
        return self
    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.
        Uses Glorot 2010 initialization assuming linear activation."""
        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"
        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            # Weight term
            initw   = self._glorot_initializer(prev_units, num_units,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('weight', initializer=initw)
            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('bias', initializer=initb)
            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias
        self.outputs.append(out)
        return self
    def add_sigmoid(self):
        """Adds a sigmoid (0,1) activation function layer to this model."""
        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            out = tf.nn.sigmoid(self.get_output())
        self.outputs.append(out)
        return self
    def add_softmax(self):
        """Adds a softmax operation to this model"""
        with tf.variable_scope(self._get_layer_str()):
            this_input = tf.square(self.get_output())
            reduction_indices = list(range(1, len(this_input.get_shape())))
            acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
            out = this_input / (acc+FLAGS.epsilon)
            #out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")
        self.outputs.append(out)
        return self
    def add_relu(self):
        """Adds a ReLU activation function to this model"""
        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.relu(self.get_output())
        self.outputs.append(out)
        return self        
    def add_elu(self):
        """Adds a ELU activation function to this model"""
        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.elu(self.get_output())
        self.outputs.append(out)
        return self
    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""
        with tf.variable_scope(self._get_layer_str()):
            t1  = .5 * (1 + leak)
            t2  = .5 * (1 - leak)
            out = t1 * self.get_output() + \
                  t2 * tf.abs(self.get_output())
        self.outputs.append(out)
        return self
    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a 2D convolutional layer."""
        assert len(self.get_output().get_shape()) == 4 and
            "Previous layer must be 4-dimensional (batch, width, height, channels)"
        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            out    = tf.nn.conv2d(self.get_output(), weight,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')
            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
        self.outputs.append(out)
        return self
    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a transposed 2D convolutional layer"""
        assert len(self.get_output().get_shape()) == 4 and 
            "Previous layer must be 4-dimensional (batch, width, height, channels)"
        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.get_output()
            output_shape = [FLAGS.batch_size,
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out    = tf.nn.conv2d_transpose(self.get_output(), weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')
            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
        self.outputs.append(out)
        return self
    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""
        assert len(self.get_output().get_shape()) == 4 and 
            "Previous layer must be 4-dimensional (batch, width, height, channels)"
        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)
        bypass = self.get_output()
        # Residual block
        for _ in range(num_layers):
            self.add_batch_norm()
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
        self.add_sum(bypass)
        return self
    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""
        assert len(self.get_output().get_shape()) == 4 and 
            "Previous layer must be 4-dimensional (batch, width, height, channels)"
        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            #bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
        bypass = self.get_output()
        # Bottleneck residual block
        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units//4, mapsize=1,       stride=1,      stddev_factor=2.)
        self.add_batch_norm()
        self.add_relu()
        if transpose:
            self.add_conv2d_transpose(num_units//4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=2.)
        else:
            self.add_conv2d(num_units//4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=2.)
        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units,    mapsize=1,       stride=1,      stddev_factor=2.)
        self.add_sum(bypass)
        return self
    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""
        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            #print("%s %s" % (prev_shape, term_shape))
            assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)
        self.outputs.append(out)
        return self
    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""
        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
        self.outputs.append(out)
        return self
    def add_upscale(self):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""
        prev_shape = self.get_output().get_shape()
        size = [2 * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(self.get_output(), size)
        self.outputs.append(out)
        return self        
    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]
    def get_variable(self, layer, name):
        """Returns a variable given its layer and name.
        The variable must already exist."""
        scope      = self._get_layer_str(layer)
        collection = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)
        # TBD: Ugly!
        for var in collection:
            if var.name[:-2] == scope+'/'+name:
                return var
        return None
    def get_all_layer_variables(self, layer):
        """Returns all variables in the given layer"""
        scope = self._get_layer_str(layer)
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)