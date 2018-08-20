import tensorflow as tf
from tensorflow.contrib.rnn import *
class RNN_model(object):
  def __init__(self, config):
    self.config = config
    self.cell_type = config['cell_type'] ##
    self.optimizer = config['optimizer'] ##
    self.hidden_units = config['hidden_size']
    self.num_layers = config['num_layers']
    self.use_residual = config['use_residual'] ##
    self.use_dropout = config['use_dropout'] ##
    self.learning_rate = config['learning_rate']
    self.max_gradient_norm = config['max_gradient_norm'] ##
    self.dtype= tf.float32 ##

    self.build_model()

  def build_model(self):
    self.init_placeholders()
    self.build_encoder()

  def init_placeholders(self):
    # encoder_inputs: [batch_size, max_time_steps, features]
    self.encoder_inputs = tf.placeholder(dtype=tf.float32,
                                         shape=(None, self.config['input_length'], self.config['input_depth']),
                                         name='encoder_inputs')

    self.decoder_targets = tf.placeholder(dtype=tf.float32,
                                         shape=(None, self.config['output_length']),
                                         name='decoder_target')
    self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')

  def build_encoder(self):
    print("building encoder..")
    with tf.variable_scope('encoder'):
      # Building encoder_cell
      self.encoder_cell = self.build_encoder_cell()
      input_layer = tf.layers.Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

      self.encoder_inputs_embedded = input_layer(self.encoder_inputs)

      self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
        cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
        dtype=self.dtype, time_major=False)

      # decoder_logits = tf.reshape(self.encoder_outputs, shape=[-1, self.encoder_outputs.shape[-1]*self.encoder_outputs.shape[-2]])
      decoder_logits = self.encoder_outputs[:,-1,:]
      self.decoder_logits = tf.layers.dense(decoder_logits, 1)
      self.loss = tf.losses.mean_squared_error(self.decoder_logits, self.decoder_targets)

      # Training summary for the current batch_loss
      tf.summary.scalar('loss', self.loss)
      self.build_optimizer()

  def build_single_cell(self):
    if (self.cell_type.lower()=='gru'):
      cell_type = GRUCell
    elif (self.cell_type.lower()=='lstm'):
      cell_type = LSTMCell
    cell = cell_type(self.hidden_units)
    if self.use_dropout:
      cell = DropoutWrapper(cell, dtype=self.dtype,
                            output_keep_prob=self.keep_prob_placeholder)
    if self.use_residual:
      cell = ResidualWrapper(cell)
    return cell

  # Building encoder cell
  def build_encoder_cell(self):
    return MultiRNNCell([self.build_single_cell() for i in xrange(self.num_layers)])

  def build_optimizer(self):
    self._global_step = tf.train.get_or_create_global_step()
    print("setting optimizer..")
    # Gradients and SGD update operation for training the model
    trainable_params = tf.trainable_variables()
    if self.optimizer.lower()=='adadelta':
      self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
    elif self.optimizer.lower()=='adam':
      self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    elif self.optimizer.lower()=='rmsprop':
      self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    else:
      self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

    # Compute gradients of loss w.r.t. all trainable variables
    gradients = tf.gradients(self.loss, trainable_params)

    # Clip gradients by a given maximum_gradient_norm
    clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

    # Update the model
    self._train_op = self.opt.apply_gradients(
      zip(clip_gradients, trainable_params), global_step=self._global_step)
