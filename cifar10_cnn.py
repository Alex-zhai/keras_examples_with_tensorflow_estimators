# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/8/24 15:31

# keras exmaple: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

from __future__ import print_function, absolute_import, division
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras import layers, models


def generate_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_shape = x_train.shape[1:]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test, x_shape


X_train, y_train, X_test, y_test, x_shape = generate_data()


def data_augmentation(img, label):
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    return img, label


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.map(data_augmentation, num_parallel_calls=100)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def eval_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def create_cnn_model(input_shape, num_classes=10):
    img_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(img_input)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out_logits = layers.Dense(num_classes)(x)
    return models.Model(inputs=img_input, outputs=out_logits)


def mem_network_fn(features, labels, mode, params):
    model = create_cnn_model(input_shape=x_shape)
    logits = model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(input=logits, axis=-1),
            "probabilities": tf.nn.softmax(logits, axis=-1),
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        eval_metric_spec = {
            'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(input=logits, axis=-1)),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    mem_network = tf.estimator.Estimator(
        model_fn=mem_network_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=100000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(), throttle_secs=120, start_delay_secs=120)
    tf.estimator.train_and_evaluate(estimator=mem_network, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("cifar10_cnn_model")


if __name__ == '__main__':
    # model = create_mem_network()
    # print(model.summary())
    tf.app.run()
