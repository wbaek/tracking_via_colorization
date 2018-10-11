from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Colorizer():
    @staticmethod
    def get(name, network_architecture, **kwargs):
        def _model_fn(features, labels, mode, params):
            """
            Args:
                features: a list of tensors, one for each tower
                labels: a list of tensors, one for each tower
                mode: ModeKeys.TRAIN or EVAL
                params: Hyperparameters suitable for tuning
            Returns:
                A EstimatorSpec object.
            """
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            data_format = params.data_format

            weight_decay = params.weight_decay
            batch_norm_decay = params.batch_norm_decay
            batch_norm_epsilon = params.batch_norm_epsilon

            optimizer = params.optimizer
            temperature = 1.0 if is_training else 0.5

            num_labels = kwargs.get('num_labels', 16)
            num_reference = kwargs.get('num_reference', 3)
            predict_backward = kwargs.get('predict_direction', 'backward') == 'backward'

            with tf.variable_scope(name, reuse=False):  # tf.AUTO_REUSE):
                with tf.name_scope('network') as name_scope:
                    model = network_architecture(
                        is_training,
                        data_format=data_format,
                        batch_norm_decay=batch_norm_decay,
                        batch_norm_epsilon=batch_norm_epsilon
                    )
                    similarity, logits, target_labels = model.forward(
                        features,
                        labels,
                        temperature=temperature,
                        num_labels=num_labels,
                        num_reference=num_reference,
                        predict_backward=predict_backward,
                    )
                    reshaped_logits = tf.reshape(logits, (-1, num_labels))
                    reshaped_target_labels = tf.reshape(target_labels, (-1,))
                    tf.logging.info('reshaped logits: %s, labels: %s', reshaped_logits.get_shape(), reshaped_target_labels.get_shape())
                    predictions = {
                        'classes': tf.argmax(input=logits, axis=-1),
                        'probabilities': tf.nn.softmax(logits, axis=-1),
                        'logits': logits,
                        'similarity': similarity,
                    }

                    weights = tf.trainable_variables()
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_logits, labels=reshaped_target_labels)
                    loss = tf.reduce_mean(loss)
                    total_loss = loss + (weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in weights]))
                    gradients = tf.gradients(total_loss, weights)

                    metrics = {
                        'metrics/accuracy': tf.metrics.accuracy(reshaped_target_labels, tf.reshape(predictions['classes'], (-1,))),
                        'metrics/loss_xentropy': tf.metrics.mean(loss),
                        'metrics/loss_total': tf.metrics.mean(total_loss),
                    }

            with tf.name_scope('summaries'):
                for weight, gradient in zip(weights, gradients):
                    variable_name = weight.name.replace(':', '_')
                    if 'BatchNorm' in variable_name:
                        continue
                    tf.summary.histogram(variable_name, weight)
                    tf.summary.histogram(variable_name + '/gradients', gradient)

            for key, value in metrics.items():
                tf.summary.scalar(key, value[1])

            with tf.name_scope('optimizer'):
                operations = [
                    optimizer.apply_gradients(zip(gradients, weights), global_step=tf.train.get_global_step()),
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)  # for batchnorm
                ]
                train_op = tf.group(*operations)

            with tf.name_scope('others'):
                tensors_to_log = {
                    'step': tf.train.get_global_step(),
                    'loss': loss,
                    'loss_total': total_loss,
                    'accuracy': tf.reduce_mean(tf.cast(tf.nn.in_top_k(reshaped_logits, reshaped_target_labels, k=1), tf.float32))
                }
                logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=kwargs.get('log_steps', 1))
                train_hooks = [logging_hook]
                predict_hooks = []

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=total_loss,
                train_op=train_op,
                training_hooks=train_hooks,
                prediction_hooks=predict_hooks,
                eval_metric_ops=metrics
            )
        return _model_fn
