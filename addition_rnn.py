# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/8/23 13:44

# keras example: https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py

'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)
Input may optionally be reversed, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
'''

from __future__ import print_function, absolute_import, division

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from sklearn.model_selection import train_test_split

DIGITS = 3
CHARS = '0123456789+ '


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """
        Initialize character table.
        :param chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char2index = dict((c, i) for i, c in enumerate(self.chars))
        self.index2char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """
        :param C: string C
        :param num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros(shape=(num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char2index[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.index2char[x] for x in x)


def generate_data(data_size, chars, digit_size=3, reverse=True):
    questions = []
    results = []
    seen_questions = set()
    MAXLEN = 2 * digit_size + 1
    while len(questions) < data_size:
        f = lambda: int(
            ''.join(np.random.choice(list('0123456789')) for _ in range(np.random.randint(1, digit_size + 1))))
        a, b = f(), f()
        # Skip any addition questions we've already seen
        # Also skip any such that x+Y == Y+x (hence the sorting).
        key = tuple(sorted((a, b)))
        if key in seen_questions:
            continue
        seen_questions.add(key)
        # Pad the data with spaces such that it is always MAXLEN.
        ques = '{}+{}'.format(a, b)
        maxlen_ques = ques + ' ' * (MAXLEN - len(ques))
        ans = str(a + b)
        # Answers can be of maximum size DIGITS + 1.
        ans += ' ' * (digit_size + 1 - len(ans))
        if reverse:
            # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
            # space used for padding.)
            maxlen_ques = maxlen_ques[::-1]
        questions.append(maxlen_ques)
        results.append(ans)
    print('Vectorization...')
    char_tables = CharacterTable(chars)
    x = np.zeros(shape=(len(questions), MAXLEN, len(chars)), dtype=np.float32)
    y = np.zeros(shape=(len(questions), digit_size + 1, len(chars)), dtype=np.float32)
    for i, sentence in enumerate(questions):
        x[i] = char_tables.encode(sentence, MAXLEN)
    for i, sentence in enumerate(results):
        y[i] = char_tables.encode(sentence, digit_size + 1)
    input_shape = (MAXLEN, len(chars))
    return x, y, input_shape


def split_data(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


x, y, input_shape = generate_data(data_size=50000, chars=CHARS, digit_size=DIGITS)
X_train, X_test, y_train, y_test = split_data(x, y)


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
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


def create_rnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(128)(inputs)
    x = layers.RepeatVector(DIGITS + 1)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(len(CHARS)))(x)
    out = layers.Activation('softmax')(x)
    return models.Model(inputs=inputs, outputs=out)


def rnn_model_fn(features, labels, mode, params):
    model = create_rnn_model(input_shape)
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
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        eval_metric_spec = {
            'accuracy': tf.metrics.accuracy(tf.argmax(labels, axis=-1), predictions=tf.argmax(input=logits, axis=-1)),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    rnn_model = tf.estimator.Estimator(
        model_fn=rnn_model_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=100000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(), throttle_secs=120, start_delay_secs=120)
    tf.estimator.train_and_evaluate(estimator=rnn_model, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("addition_rnn_model")


if __name__ == '__main__':
    # model = create_rnn_model(input_shape)
    # print(model.summary())
    tf.app.run()
