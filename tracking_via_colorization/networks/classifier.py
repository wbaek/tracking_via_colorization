from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Classifier():
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

            with tf.variable_scope(name, reuse=False):  # tf.AUTO_REUSE):
                with tf.name_scope('network') as name_scope:
                    model = network_architecture(
                        is_training,
                        data_format=data_format,
                        batch_norm_decay=batch_norm_decay,
                        batch_norm_epsilon=batch_norm_epsilon
                    )
                    logits = model.forward(features, **kwargs)
                    predictions = {
                        'classes': tf.argmax(input=logits, axis=1),
                        'probabilities': tf.nn.softmax(logits),
                        'logits': logits
                    }
                    metrics = {
                        'accuracy': tf.metrics.accuracy(labels, predictions['classes'])
                    }

                    weights = tf.trainable_variables()
                    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
                    loss = tf.reduce_mean(loss)
                    loss += weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in weights])
                    gradients = tf.gradients(loss, weights)
            if is_training:
                tf.summary.scalar("accuracy", metrics['accuracy'][1])

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
                    'accuracy': tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, k=1), tf.float32))
                }
                logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=kwargs.get('log_steps', 1))
                train_hooks = [logging_hook]
                predict_hooks = []

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
                prediction_hooks=predict_hooks,
                eval_metric_ops=metrics
            )
        return _model_fn
