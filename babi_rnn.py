# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/8/23 17:59

# keras exmaple: https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py

from __future__ import print_function, absolute_import, division
from functools import reduce
import tensorflow as tf
import numpy as np
import re
import tarfile
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    """
    :param lines:
    :param only_supporting: If only_supporting is true, only the sentences that support the answer are kept.
    :return:
    """
    data = []
    story = []
    for line in lines:
        # line = line.decode('utf-8').strip()
        # 19 Where is the football?       hallway 15 17   nid, question, answer, support ids
        # 20 John moved to the hallway.         nid, story
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            # start another a new story
            story = []
        if '\t' in line:
            q, a, support_ids = line.split('\t')
            q = tokenize(q)
            if only_supporting:
                # Only select the related substory
                support_ids = map(int, support_ids.split())
                support_story = [story[id - 1] for id in support_ids]
            else:
                support_story = [x for x in story if x]
            data.append((support_story, q, a))
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    """
    Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, ans) for story, q, ans in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    vec_sqa = (pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.asarray(ys))
    return vec_sqa


def generate_data():
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
    challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    with tarfile.open(path) as tar:
        train = get_stories(tar.extractfile(challenge.format('train')))
        test = get_stories(tar.extractfile(challenge.format('test')))
    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])
    # 36 words
    vocab = sorted(vocab)
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))
    # xs:1000 * 552  xqs: 1000*5  ys: 1000*36
    train_sqa_vec = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    test_sqa_vec = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)
    return train_sqa_vec, test_sqa_vec, vocab_size, story_maxlen, query_maxlen


train_sqa_vec, test_sqa_vec, vocab_size, story_maxlen, query_maxlen = generate_data()


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"story": train_sqa_vec[0], "ques": train_sqa_vec[1]}, train_sqa_vec[2]))
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def eval_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"story": test_sqa_vec[0], "ques": test_sqa_vec[1]}, test_sqa_vec[2]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def create_mem_network():
    sentence = layers.Input(shape=(story_maxlen,), dtype=tf.int32)
    encoded_sentence = layers.Embedding(input_dim=vocab_size, output_dim=50)(sentence)
    encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

    question = layers.Input(shape=(query_maxlen,), dtype=tf.int32)
    encoded_ques = layers.Embedding(input_dim=vocab_size, output_dim=50)(question)
    encoded_ques = layers.Dropout(0.3)(encoded_ques)
    encoded_ques = layers.LSTM(50)(encoded_ques)
    encoded_ques = layers.RepeatVector(story_maxlen)(encoded_ques)

    merged = layers.add([encoded_sentence, encoded_ques])
    merged = layers.LSTM(50)(merged)
    merged = layers.Dropout(0.3)(merged)
    preds = layers.Dense(vocab_size, activation=None)(merged)
    return models.Model(inputs=[sentence, question], outputs=preds)


def mem_network_fn(features, labels, mode, params):
    model = create_mem_network()
    story = features['story']
    ques = features['ques']
    logits = model([story, ques])
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
    mem_network = tf.estimator.Estimator(
        model_fn=mem_network_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=100000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator=mem_network, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("babi_rnn_model")


if __name__ == '__main__':
    # model = create_mem_network()
    # print(model.summary())
    tf.app.run()
