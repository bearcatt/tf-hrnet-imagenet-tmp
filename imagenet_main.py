# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a HRNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf

import distribution_utils
import hrnet_model
import imagenet_preprocessing

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NUM_IMAGES = {
  'train': 1281167,
  'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000


def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
      os.path.join(data_dir, 'train-%05d-of-01024' % i)
      for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
      os.path.join(data_dir, 'validation-%05d-of-00128' % i)
      for i in range(128)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

  image/height: 462
  image/width: 581
  image/colorspace: 'RGB'
  image/channels: 3
  image/class/label: 615
  image/class/synset: 'n03623198'
  image/class/text: 'knee pad'
  image/object/bbox/xmin: 0.1
  image/object/bbox/xmax: 0.9
  image/object/bbox/ymin: 0.2
  image/object/bbox/ymax: 0.6
  image/object/bbox/label: 615
  image/format: 'JPEG'
  image/filename: 'ILSVRC2012_val_00041207.JPEG'
  image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
    'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                           default_value=''),
    'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                               default_value=-1),
    'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
    {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                 'image/object/bbox/ymin',
                                 'image/object/bbox/xmax',
                                 'image/object/bbox/ymax']})

  features = tf.io.parse_single_example(serialized=example_serialized,
                                        features=feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

  return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
  raw_record: scalar Tensor tf.string containing a serialized
    Example protocol buffer.
  is_training: A boolean denoting whether the input is for training.
  dtype: data type to use for images/features.

  Returns:
  Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = _parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image(
    image_buffer=image_buffer,
    bbox=bbox,
    output_height=DEFAULT_IMAGE_SIZE,
    output_width=DEFAULT_IMAGE_SIZE,
    num_channels=NUM_CHANNELS,
    is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           drop_remainder=False,
                           tf_data_experimental_slack=False):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
  dataset: A Dataset representing raw records
  is_training: A boolean denoting whether the input is for training.
  batch_size: The number of samples per batch.
  shuffle_buffer: The buffer size to use when shuffling records. A larger
    value results in better randomness, but smaller values reduce startup
    time and use less memory.
  parse_record_fn: A function that takes a raw record and returns the
    corresponding (image, label) pair.
  num_epochs: The number of epochs to repeat the dataset.
  dtype: Data type to use for images/features.
  datasets_num_private_threads: Number of threads for a private
    threadpool created for all datasets computation.
  drop_remainder: A boolean indicates whether to drop the remainder of the
    batches. If True, the batch dimension will be static.
  tf_data_experimental_slack: Whether to enable tf.data's
    `experimental_slack` option.

  Returns:
  Dataset of (image, label) pairs ready for iteration.
  """
  # Defines a specific size thread pool for tf.data operations.
  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (
      datasets_num_private_threads)
    dataset = dataset.with_options(options)
    tf.compat.v1.logging.info('datasets_num_private_threads: %s',
                              datasets_num_private_threads)

  # Disable intra-op parallelism to optimize for throughput instead of latency.
  options = tf.data.Options()
  options.experimental_threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)

  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # Repeats the dataset for the number of epochs to train.
  dataset = dataset.repeat(num_epochs)

  # Parses the raw records into images and labels.
  dataset = dataset.map(
    lambda value: parse_record_fn(value, is_training, dtype),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  if tf_data_experimental_slack:
    options = tf.data.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)

  return dataset


def input_fn(is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False,
             tf_data_experimental_slack=False):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    parse_record_fn: Function to use for parsing the records.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if input_context:
    tf.compat.v1.logging.info(
      'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
        input_context.input_pipeline_id, input_context.num_input_pipelines))
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means that up to 10 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
    tf.data.TFRecordDataset,
    cycle_length=10,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return process_record_dataset(
    dataset=dataset,
    is_training=is_training,
    batch_size=batch_size,
    shuffle_buffer=_SHUFFLE_BUFFER,
    parse_record_fn=parse_record_fn,
    num_epochs=num_epochs,
    dtype=dtype,
    datasets_num_private_threads=datasets_num_private_threads,
    drop_remainder=drop_remainder,
    tf_data_experimental_slack=tf_data_experimental_slack,
  )


class ImagenetModel(hrnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, hrnet_size, 
               data_format=None, 
               num_classes=NUM_CLASSES, 
               dtype=tf.float32):
    """These are the parameters that work for Imagenet data.

    Args:
      hrnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      dtype: The Tensorflow dtype to use for calculations.
    """
    super(ImagenetModel, self).__init__(
      hrnet_size=hrnet_size,
      bottleneck=False,
      num_classes=num_classes,
      num_filters=64,
      kernel_size=3,
      conv_stride=2,
      module_sizes=[1, 1, 4, 3],
      block_sizes=[4, 4, 4, 4],
      base_channel=128,
      final_conv_channel=2048,
      data_format=data_format,
      dtype=dtype
    )


def learning_rate_with_decay(batch_size, batch_denom, num_images,
                             boundary_epochs, decay_rates,
                             base_lr=0.1, warmup=False):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, vals)
    if warmup:
      warmup_steps = int(batches_per_epoch * 5)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
        warmup_steps, tf.float32))
      return tf.cond(pred=global_step < warmup_steps,
                     true_fn=lambda: warmup_lr,
                     false_fn=lambda: lr)
    return lr

  return learning_rate_fn


def hrnet_model_fn(features, labels, mode, model_class,
                   hrnet_size, weight_decay, learning_rate_fn,
                   momentum, data_format, loss_scale,
                   loss_filter_fn=None, dtype=tf.float32,
                   label_smoothing=0.0):
  """Shared functionality for different hrnet model_fns.

  Initializes the HRNetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of HRNetModel.
    hrnet_size: A single integer for the size of the HRNetModel model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.
    label_smoothing: If greater than 0 then smooth the labels.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Checks that features/images have same data type being used for calculations.
  assert features.dtype == dtype

  model = model_class(hrnet_size, data_format, dtype=dtype)
  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)
  predictions = {
    'classes': tf.argmax(input=logits, axis=1),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  if label_smoothing != 0.0:
    one_hot_labels = tf.one_hot(labels, NUM_CLASSES)
    cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=one_hot_labels,
      label_smoothing=label_smoothing)
  else:
    cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name

  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
    # loss is computed using fp32 for numerical stability.
    [
      tf.nn.l2_loss(tf.cast(v, tf.float32))
      for v in tf.compat.v1.trainable_variables()
      if loss_filter_fn(v.name)
    ])
  tf.compat.v1.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.compat.v1.train.MomentumOptimizer(
      learning_rate=learning_rate, momentum=momentum)

    fp16_implementation = getattr(FLAGS, 'fp16_implementation', None)
    if fp16_implementation == 'graph_rewrite':
      optimizer = (
        tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer, loss_scale=loss_scale))

    if loss_scale != 1 and fp16_implementation != 'graph_rewrite':
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      grad_vars = optimizer.compute_gradients(loss)
      minimize_op = optimizer.apply_gradients(grad_vars, global_step)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

  accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'])
  accuracy_top_5 = tf.compat.v1.metrics.mean(
    tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))
  metrics = {'accuracy': accuracy,
             'accuracy_top_5': accuracy_top_5}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
  tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
  tf.compat.v1.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                    loss=loss, train_op=train_op, eval_metric_ops=metrics)


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for HRNet to be used with our Estimator."""

  learning_rate_fn = learning_rate_with_decay(
    batch_size=params['batch_size'] * params.get('num_workers', 1),
    batch_denom=256, num_images=NUM_IMAGES['train'],
    boundary_epochs=[30, 60, 90], decay_rates=[1, 0.1, 0.01, 0.001],
    warmup=True, base_lr=0.05)

  return hrnet_model_fn(
    features=features,
    labels=labels,
    mode=mode,
    model_class=ImagenetModel,
    hrnet_size=params['hrnet_size'],
    weight_decay=FLAGS.weight_decay,
    learning_rate_fn=learning_rate_fn,
    momentum=0.9,
    data_format=params['data_format'],
    loss_scale=params['loss_scale'],
    loss_filter_fn=None,
    dtype=params['dtype'],
    label_smoothing=FLAGS.label_smoothing
  )


def hrnet_main(model_function, input_function):
  """Shared main loop for ResNet Models.

  Args:
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
  """

  # Configures cluster spec for distribution strategy.
  num_workers = distribution_utils.configure_cluster(FLAGS.worker_hosts,
                                                     FLAGS.task_index)

  # Creates session config. allow_soft_placement = True, is required for
  # multi-GPU and is not harmful for other modes.
  session_config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
    intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
    allow_soft_placement=True)
  session_config.gpu_options.allow_growth = True

  distribution_strategy = distribution_utils.get_distribution_strategy(
    distribution_strategy=FLAGS.distribution_strategy,
    num_gpus=FLAGS.num_gpus,
    num_workers=num_workers,
    all_reduce_alg=FLAGS.all_reduce_alg,
    num_packs=FLAGS.num_packs)

  # Creates a `RunConfig` that checkpoints every half an hour.
  run_config = tf.estimator.RunConfig(
    train_distribute=distribution_strategy,
    session_config=session_config,
    save_checkpoints_secs=60 * 30,
    save_checkpoints_steps=None)

  # Initializes from pretrained model.
  if FLAGS.pretrained_model_checkpoint_path is not None:
    warm_start_settings = tf.estimator.WarmStartSettings(
      FLAGS.pretrained_model_checkpoint_path)
  else:
    warm_start_settings = None

  if FLAGS.dtype == 'fp32':
    dtype = tf.float32
  elif FLAGS.dtype == 'fp316':
    dtype = tf.float16

  classifier = tf.estimator.Estimator(
    model_fn=model_function, model_dir=FLAGS.model_dir, config=run_config,
    warm_start_from=warm_start_settings, params={
      'hrnet_size': int(FLAGS.hrnet_size),
      'data_format': FLAGS.data_format,
      'batch_size': FLAGS.batch_size,
      'loss_scale': FLAGS.loss_scale,
      'dtype': dtype,
      'num_workers': num_workers,
    })

  def input_fn_train(num_epochs, input_context=None):
    return input_function(
      is_training=True,
      data_dir=FLAGS.data_dir,
      batch_size=distribution_utils.per_replica_batch_size(
        FLAGS.batch_size, FLAGS.num_gpus),
      num_epochs=num_epochs,
      dtype=dtype,
      datasets_num_private_threads=FLAGS.datasets_num_private_threads,
      input_context=input_context)

  def input_fn_eval():
    return input_function(
      is_training=False,
      data_dir=FLAGS.data_dir,
      batch_size=distribution_utils.per_replica_batch_size(
        FLAGS.batch_size, FLAGS.num_gpus),
      num_epochs=1,
      dtype=dtype)

  if FLAGS.eval_only or not FLAGS.train_epochs:
    tf.compat.v1.logging.info('Starting to evaluate.')
    eval_results = classifier.evaluate(input_fn=input_fn_eval, 
      checkpoint_path=FLAGS.pretrained_model_checkpoint_path)

    # TODO: put it into logs
    print(eval_results)
  else:
    tf.compat.v1.logging.info('Starting to train.')
    classifier.train(
      input_fn=lambda input_context=None: input_fn_train(
        FLAGS.train_epochs, input_context=input_context))


if __name__ == "__main__":
  FLAGS = tf.app.flags.FLAGS

  tf.flags.DEFINE_float(
    name='weight_decay', default=1e-4,
    help="Weight decay coefficiant for l2 regularization.")
  tf.flags.DEFINE_float(
    name='label_smoothing', default=0.0,
    help="Label smoothing parameter used in the softmax_cross_entropy")
  tf.app.flags.DEFINE_string(
    name='worker_hosts', default=None,
    help="Comma-separated list of worker ip:port pairs for running "
         "multi-worker models with DistributionStrategy.  The user would "
         "start the program on each host with identical value for this flag.")
  tf.flags.DEFINE_integer(
    name='task_index', default=-1,
    help="If multi-worker training, the task_index of this worker.")
  tf.flags.DEFINE_integer(
    name='inter_op_parallelism_threads', short_name='inter', default=0,
    help="Number of inter_op_parallelism_threads to use for CPU. "
         "See TensorFlow config.proto for details.")
  tf.flags.DEFINE_integer(
    name='intra_op_parallelism_threads', short_name='intra', default=0,
    help="Number of intra_op_parallelism_threads to use for CPU. "
         "See TensorFlow config.proto for details.")
  tf.flags.DEFINE_string(
    name='distribution_strategy', short_name='ds', default='default',
    help="The Distribution Strategy to use for training. "
         "Accepted values are 'off', 'default', 'one_device', "
         "'mirrored', 'parameter_server', 'collective', "
         "case insensitive. 'off' means not to use "
         "Distribution Strategy; 'default' means to choose "
         "from `MirroredStrategy` or `OneDeviceStrategy` "
         "according to the number of GPUs.")
  tf.flags.DEFINE_integer(
    name='num_gpus', short_name='ng', default=8,
    help="How many GPUs to use at each worker with the "
         "DistributionStrategies API. The default is 8.")
  tf.flags.DEFINE_string(
    name='all_reduce_alg', short_name='ara', default=None,
    help="Defines the algorithm to use for performing all-reduce."
         "When specified with MirroredStrategy for single "
         "worker, this controls "
         "tf.contrib.distribute.AllReduceCrossTowerOps.  When "
         "specified with MultiWorkerMirroredStrategy, this "
         "controls "
         "tf.distribute.experimental.CollectiveCommunication; "
         "valid options are `ring` and `nccl`.")
  tf.flags.DEFINE_integer(
    name='num_packs', default=1,
    help="Sets `num_packs` in the cross device ops used in "
         "MirroredStrategy.  For details, see "
         "tf.distribute.NcclAllReduce.")
  tf.flags.DEFINE_string(
    name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
    help="If not None initialize all the network except the final layer with "
         "these values")
  tf.flags.DEFINE_string(
    name='model_dir', short_name='md', default='checkpoints',
    help="The location of the model checkpoint files.")
  tf.flags.DEFINE_integer(
    name='hrnet_size', default=32,
    help="The width of hrnet.")
  tf.flags.DEFINE_enum(
    name='data_format', short_name='df', default='channels_first',
    enum_values=['channels_first', 'channels_last'],
    help="A flag to override the data format used in the model. "
         "channels_first provides a performance boost on GPU but is not "
         "always compatible with CPU. If left unspecified, the data format "
         "will be chosen automatically based on whether TensorFlow was "
         "built for CPU or GPU.")
  tf.flags.DEFINE_integer(
    name='batch_size', short_name='bs', default=256,
    help="Batch size for training and evaluation. When using "
         "multiple gpus, this is the global batch size for "
         "all devices. For example, if the batch size is 32 "
         "and there are 4 GPUs, each GPU will get 8 examples on "
         "each step.")
  tf.flags.DEFINE_integer(
    name='loss_scale', short_name='ls', default=1,
    help="The amount to scale the loss by when the model is run. Before "
         "gradients are computed, the loss is multiplied by the loss scale, "
         "making all gradients loss_scale times larger. To adjust for this, "
         "gradients are divided by the loss scale before being applied to "
         "variables. This is mathematically equivalent to training without "
         "a loss scale, but the loss scale helps avoid some intermediate "
         "gradients from underflowing to zero. If not provided the default "
         "for fp16 is 128 and 1 for all other dtypes.")
  tf.flags.DEFINE_enum(
    name='dtype', short_name='dt', default='fp32',
    enum_values=['fp16', 'fp32'],
    help="The TensorFlow datatype used for calculations. "
         "Variables may be cast to a higher precision on a "
         "case-by-case basis for numerical stability.")
  # TODO: test
  tf.flags.DEFINE_enum(
      name='fp16_implementation', default='None',
      enum_values=('casting', 'graph_rewrite', 'None'),
      help="When --dtype=fp16, how fp16 should be implemented. This has no "
           "impact on correctness. 'casting' will cause manual tf.casts to "
           "be inserted in the model. 'graph_rewrite' means "
           "tf.train.experimental.enable_mixed_precision_graph_rewrite will "
           "be used to automatically use fp16 without any manual casts.")
  tf.flags.DEFINE_integer(
    name="train_epochs", short_name="te", default=100,
    help="The number of epochs used to train.")
  tf.flags.DEFINE_string(
    name="data_dir", short_name="dd", default="dataset",
    help="The location of the input data.")
  tf.flags.DEFINE_integer(
    name="datasets_num_private_threads", default=8,
    help="Number of threads for a private threadpool created for all"
         "datasets computation.")
  tf.flags.DEFINE_boolean(
    name='eval_only', default=False,
    help="Skip training and only perform evaluation on "
         "the latest checkpoint.")

  hrnet_main(imagenet_model_fn, input_fn)
