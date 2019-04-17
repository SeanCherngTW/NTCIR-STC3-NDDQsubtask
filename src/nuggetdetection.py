import os
import param
import timeit
import logging
import datahelper
import stctokenizer
import tensorflow as tf
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from gensim.models import word2vec
from time import gmtime, strftime, localtime
from gensim.models.keyedvectors import KeyedVectors

doclen = param.doclen
embsize = param.embsize
max_sent = param.max_sent
NDclasses = param.NDclasses

logger = logging.getLogger('ND task')
tf.logging.set_verbosity(tf.logging.ERROR)


def weight_variable(shape, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.get_variable(
            shape=shape,
            initializer=tf.contrib.keras.initializers.he_normal(),
            name=name,
        )


def bias_variable(shape, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.Variable(tf.constant(0.1, shape=shape, name=name))


def conv2d(x, W, embsize, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.nn.conv2d(x, W, strides=[1, 1, embsize, 1], padding='SAME', name=name)


def conv1d(x, filter_size, num_filter, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.layers.conv1d(
            inputs=x,
            filters=num_filter,
            kernel_size=filter_size,
            activation=tf.nn.relu,
            padding='SAME',
            name=name,
        )


def maxpool(h, pool_size, strides, name, reuse=False):
    with tf.variable_scope("", reuse=reuse):
        return tf.layers.max_pooling1d(
            inputs=h,
            pool_size=pool_size,
            strides=strides,
            padding='VALID',
            name=name,
        )


def init_input(doclen, embsize):
    # doclen = 150, embsize = 256
    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, max_sent, doclen, embsize], name='input_X')
        y = tf.placeholder(tf.float32, [None, max_sent, NDclasses], name='output_Y')
        bs = tf.placeholder(tf.int32, [], name='batch_size')
        turns = tf.placeholder(tf.int32, [None, ], name='turns')
        masks = tf.placeholder(tf.float32, [None, max_sent, NDclasses], name='masks')
        num_sent = tf.placeholder(tf.int32, [], name='num_sent')
    return x, y, bs, turns, masks, num_sent


def loss_function(pred, y, batch_size, num_sent, masks):
    with tf.name_scope('Loss'):
        cost = 0
        # pred_masked = tf.multiply(pred, masks)
        # pred_sents = tf.unstack(pred_masked, axis=1)
        pred_sents = tf.unstack(pred, axis=1)
        y_sents = tf.unstack(y, axis=1)
        num_sent = tf.cast(num_sent, tf.float32)

        for pred_sent, y_sent in zip(pred_sents, y_sents):  # (?, 7) * 7

            logger.debug('Loss: pred_sent.shape = {}'.format(str(pred_sent.shape)))
            logger.debug('Loss: y_sent.shape = {}'.format(str(y_sent.shape)))
            pred_sent = tf.reshape(pred_sent, [-1, NDclasses])
            y_sent = tf.reshape(y_sent, [-1, NDclasses])

            cost += -tf.reduce_sum(y_sent * tf.log(tf.clip_by_value(pred_sent, 1e-10, 1.0)))

    return tf.divide(cost, num_sent)


def build_multistackCNN(x_split, bs, filter_size, num_filters, gating, batch_norm):
    is_first = True
    sentCNNs_reuse = False

    with tf.name_scope('SentCNN'):
        for i, x_sent in enumerate(x_split):
            # print('CNN input shape', x_sent.shape)

            if i % 2 == 0:  # customer
                speaker = tf.fill((bs, 1, 1), 0.0)
            else:  # helpdesk
                speaker = tf.fill((bs, 1, 1), 1.0)

            if gating:
                for layer, Fnum in enumerate(num_filters, 1):
                    sentCNN_convA = conv1d(x_sent, filter_size[0], Fnum, 'sentCNN_convA{}'.format(layer), sentCNNs_reuse)
                    sentCNN_convB = conv1d(x_sent, filter_size[1], Fnum, 'sentCNN_convB{}'.format(layer), sentCNNs_reuse)
                    x_sent = tf.multiply(sentCNN_convA, tf.nn.sigmoid(sentCNN_convB), name='gating{}'.format(layer))
                    if batch_norm:
                        x_sent = tf.layers.batch_normalization(x_sent)

                sentCNN_pool = maxpool(x_sent, doclen, 1, 'sentCNN_pool', sentCNNs_reuse)
                concated = tf.concat([sentCNN_pool, speaker], axis=-1, name='sentCNN_concated')

            else:
                sentCNN_convA = x_sent
                sentCNN_convB = x_sent
                for layer, Fnum in enumerate(num_filters, 1):
                    sentCNN_convA = conv1d(sentCNN_convA, filter_size[0],
                                           Fnum, 'sentCNN_convA{}'.format(layer), sentCNNs_reuse)
                    sentCNN_convB = conv1d(sentCNN_convB, filter_size[1],
                                           Fnum, 'sentCNN_convB{}'.format(layer), sentCNNs_reuse)
                    if batch_norm:
                        sentCNN_convA = tf.layers.batch_normalization(sentCNN_convA)
                        sentCNN_convB = tf.layers.batch_normalization(sentCNN_convB)

                logger.debug('sentCNN before maxpool {}'.format(str(sentCNN_convA.shape)))
                sentCNN_poolA = maxpool(sentCNN_convA, doclen, 1, 'sentCNN_poolA', sentCNNs_reuse)
                sentCNN_poolB = maxpool(sentCNN_convB, doclen, 1, 'sentCNN_poolB', sentCNNs_reuse)
                concated = tf.concat([sentCNN_poolA, sentCNN_poolB, speaker], axis=-1, name='sentCNN_concated')
                logger.debug('sentCNN after maxpool {}'.format(str(sentCNN_poolA.shape)))

            if is_first:
                sentCNNs = concated
                sentCNNs_reuse = True
                is_first = False
            else:
                sentCNNs = tf.concat([sentCNNs, concated], axis=1, name='sentCNN_concate_sent')
    return sentCNNs


def build_RNN(sentCNNs, bs, turns, rnn_hiddens, batch_norm, name, rnn_type, keep_prob, num_layers):
    def _get_cell(rnn_type, rnn_hiddens):
        assert rnn_type in ['Bi-LSTM', 'Bi-GRU']
        if rnn_type == 'Bi-LSTM':
            return tf.contrib.rnn.BasicLSTMCell(rnn_hiddens, forget_bias=1.0)
        else:
            return tf.contrib.rnn.GRUCell(rnn_hiddens)

    fw_cells = []
    bw_cells = []
    with tf.name_scope(name):
        with tf.name_scope(rnn_type):
            for _ in range(num_layers):
                fw_cell = tf.contrib.rnn.DropoutWrapper(
                    _get_cell(rnn_type, rnn_hiddens),
                    input_keep_prob=keep_prob,
                    output_keep_prob=keep_prob,
                )
                bw_cell = tf.contrib.rnn.DropoutWrapper(
                    _get_cell(rnn_type, rnn_hiddens),
                    input_keep_prob=keep_prob,
                    output_keep_prob=keep_prob,
                )

                fw_cells.append(fw_cell)
                bw_cells.append(bw_cell)

        fw_cells = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_cells = tf.contrib.rnn.MultiRNNCell(bw_cells)
        init_state_fw = fw_cells.zero_state(bs, tf.float32)
        init_state_bw = bw_cells.zero_state(bs, tf.float32)

        (output_fw, output_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cells,
            cell_bw=bw_cells,
            inputs=sentCNNs,
            sequence_length=turns,
            initial_state_fw=init_state_fw,
            initial_state_bw=init_state_bw,
            time_major=False,
            scope=name,
        )

        with tf.name_scope('add_Fw_Bw'):
            # rnn_output = tf.concat([output_fw, output_bw], axis=-1)
            rnn_output = tf.nn.tanh(tf.add(output_fw, output_bw))
            logger.debug('{} rnn_output {}'.format(name, str(rnn_output.shape)))

    return rnn_output


def memory_enhanced(rnn_output, input_memory, output_memory):
    with tf.name_scope('memory_enhanced'):
        is_first = True
        rnn_output = tf.unstack(rnn_output, axis=1)
        input_memory = tf.unstack(input_memory, axis=1)
        for sent_t in rnn_output:  # for sentence in time t
            attention_at_time_t = []
            for context_i in input_memory:  # for context sentence i
                # sent_t = (?, 1024), context_i = (?, 1024), _attention = (?, )
                _attention = tf.reduce_sum(tf.multiply(sent_t, context_i), axis=1)
                attention_at_time_t.append(_attention)

            # attention_at_time_t = 7 * (?, ) --stack--> (?, 7), attention_weight_at_time_t = (?, 7)
            # attention_weight_at_time_t = tf.nn.tanh(tf.stack(attention_at_time_t, axis=1))
            attention_weight_at_time_t = tf.nn.softmax(tf.stack(attention_at_time_t, axis=1))
            attention_weight_at_time_t = tf.reshape(attention_weight_at_time_t, [-1, max_sent, 1])  # boardcast weight

            # attention_weight_at_time_t = (?, 7), output_memory = (?, 7, 1024), weighted_output_memory = (?, 7, 1024)
            logger.debug('attention_weight_at_time_t {}'.format(str(attention_weight_at_time_t.shape)))
            logger.debug('output_memory {}'.format(str(output_memory.shape)))
            weighted_output_memory = tf.multiply(output_memory, attention_weight_at_time_t)
            logger.debug('weighted_output_memory {}'.format(str(weighted_output_memory.shape)))

            # weighted_output_memory = (?, 7, 1024), weighted_sum = (?, 1024)
            weighted_sum = tf.reduce_sum(weighted_output_memory, axis=1)

            # sent_t_with_memory = (?, 1024)
            sent_t_with_memory = tf.add(weighted_sum, sent_t)
            sent_t_with_memory = tf.expand_dims(sent_t_with_memory, axis=1)

            if is_first:
                sents_with_memory = sent_t_with_memory
                is_first = False
            else:
                sents_with_memory = tf.concat([sents_with_memory, sent_t_with_memory], axis=1)

    return sents_with_memory  # (?, 7, 1024)


def build_FC(bs, rnn_outputs, rnn_hiddens, batch_norm, masks, keep_prob):
    logger.debug('FC Input {}'.format(str(rnn_outputs.shape)))
    rnn_outputs = tf.unstack(rnn_outputs, axis=1)
    prev_nd = tf.fill((bs, max_sent), 0.0)
    logger.debug('masks {}'.format(str(masks.shape)))
    masks = tf.unstack(masks, max_sent, axis=1)
    fc_outputs = []
    with tf.name_scope('FCLayer'):
        is_first = True
        fc_reuse = False

        logger.debug('FC Input per sent {}'.format(str(rnn_outputs[0].shape)))

        for i, rnn_output in enumerate(rnn_outputs):
            if batch_norm:
                rnn_output = tf.layers.batch_normalization(rnn_output)

            if i % 2 == 0:  # customer
                speaker_mask = tf.concat([tf.ones([bs, 4], dtype=tf.float32), tf.zeros([bs, 3], dtype=tf.float32)], axis=1)
            else:  # helpdesk
                speaker_mask = tf.concat([tf.zeros([bs, 4], dtype=tf.float32), tf.ones([bs, 3], dtype=tf.float32)], axis=1)

            rnn_output = tf.nn.dropout(rnn_output, keep_prob)

            fc1_W = weight_variable([rnn_output.shape[1], NDclasses], name='fc1_W', reuse=fc_reuse)
            fc1_b = bias_variable([NDclasses, ], name='fc1_b', reuse=fc_reuse)
            fc1_out = tf.matmul(rnn_output, fc1_W) + fc1_b
            y_pre = tf.nn.softmax(fc1_out)  # (?, 7)

            # y_pre = tf.multiply(y_pre, speaker_mask)

            y_pre = tf.expand_dims(y_pre, axis=1)

            if is_first:
                fc_outputs = y_pre
                fc_reuse = True
                is_first = False
            else:
                fc_outputs = tf.concat([fc_outputs, y_pre], axis=1)

            prev_nd = y_pre

    return fc_outputs


def CNNRNN(x, y, bs, turns, keep_prob, rnn_hiddens, filter_size, num_filters, gating, batch_norm, num_layers, masks, memory_rnn_type=None):

    print('x.shape', x.shape)
    # x_split = tf.split(x, max_sent, axis=1)
    x_split = tf.unstack(x, axis=1)
    sentCNNs = build_multistackCNN(x_split, bs, filter_size, num_filters, gating, batch_norm)  # Sentence representation
    logger.debug('sentCNNs output {}'.format(str(sentCNNs.shape)))
    rnn_output = build_RNN(sentCNNs, bs, turns, rnn_hiddens, batch_norm, 'context_RNN', 'Bi-LSTM', keep_prob, num_layers)  # Sentence context
    logger.debug('rnn_output input {}'.format(str(rnn_output.shape)))

    # Memory enhanced structure
    if memory_rnn_type:
        input_memory = build_RNN(rnn_output, bs, turns, rnn_hiddens, batch_norm, 'input_memory', memory_rnn_type, keep_prob, 1)
        output_memory = build_RNN(rnn_output, bs, turns, rnn_hiddens, batch_norm, 'output_memory', memory_rnn_type, keep_prob, 1)
        rnn_output = memory_enhanced(rnn_output, input_memory, output_memory)

    fc_outputs = build_FC(bs, rnn_output, rnn_hiddens, batch_norm, masks, keep_prob)
    # viterbi_sequence, viterbi_score = build_CRF(fc_outputs, y, turns)

    # return viterbi_score
    return fc_outputs


def CNNCNN(x, y, bs, turns, keep_prob, fc_hiddens, filter_size, num_filters, gating, batch_norm, masks):

    x_split = tf.unstack(x, axis=1)

    sentCNNs_reuse = False
    is_first = True
    sentCNNs = []

    # Sentence CNNs
    for i, x_sent in enumerate(x_split):
        if i % 2 == 0:  # customer
            speaker = tf.fill((bs, 1, 1), 0.0)
        else:  # helpdesk
            speaker = tf.fill((bs, 1, 1), 1.0)

        if gating:
            for layer, Fnum in enumerate(num_filters):
                sentCNN_convA = conv1d(x_sent, filter_size[0], Fnum, 'sentCNN_convA{}'.format(layer), sentCNNs_reuse)
                sentCNN_convB = conv1d(x_sent, filter_size[1], Fnum, 'sentCNN_convB{}'.format(layer), sentCNNs_reuse)
                x_sent = tf.multiply(sentCNN_convA, tf.nn.sigmoid(sentCNN_convB), name='gating{}'.format(layer))
                if batch_norm:
                    x_sent = tf.layers.batch_normalization(x_sent)

            sentCNN_pool = maxpool(x_sent, doclen, 1, 'sentCNN_pool', sentCNNs_reuse)
            concated = tf.concat([sentCNN_pool, speaker], axis=-1)

        else:
            sentCNN_convA = x_sent
            sentCNN_convB = x_sent
            for layer, Fnum in enumerate(num_filters):
                sentCNN_convA = conv1d(sentCNN_convA, filter_size[0], Fnum, 'sentCNN_convA{}'.format(layer), sentCNNs_reuse)
                sentCNN_convB = conv1d(sentCNN_convB, filter_size[1], Fnum, 'sentCNN_convB{}'.format(layer), sentCNNs_reuse)
                if batch_norm:
                    sentCNN_convA = tf.layers.batch_normalization(sentCNN_convA)
                    sentCNN_convB = tf.layers.batch_normalization(sentCNN_convB)

            sentCNN_poolA = maxpool(sentCNN_convA, doclen, 1, 'sentCNN_poolA', sentCNNs_reuse)
            sentCNN_poolB = maxpool(sentCNN_convB, doclen, 1, 'sentCNN_poolB', sentCNNs_reuse)
            concated = tf.concat([sentCNN_poolA, sentCNN_poolB, speaker], axis=-1)

        if is_first:
            sentCNNs_reuse = True
            is_first = False

        sentCNNs.append(concated)

    # Prepare Context CNN
    sentCNN_shape = sentCNNs[0].shape
    contextCNNs = []
    contextCNNs_reuse = False
    is_first = True
    for i in range(max_sent):
        if i == 0:
            start = tf.fill((bs, 1, sentCNN_shape[-1]), 0.0)  # start補零
            contextCNNs.append(tf.concat([start, sentCNNs[i], sentCNNs[i + 1]], axis=-1))
        elif i == max_sent - 1:
            end = tf.fill((bs, 1, sentCNN_shape[-1]), 0.0)  # end 補零
            contextCNNs.append(tf.concat([sentCNNs[i - 1], sentCNNs[i], end], axis=-1))
        else:
            contextCNNs.append(tf.concat([sentCNNs[i - 1], sentCNNs[i], sentCNNs[i + 1]], axis=-1))

    # Context CNNs
    for i, x_context in enumerate(contextCNNs):
        if gating:
            for layer, Fnum in enumerate(num_filters):
                contextCNN_convA = conv1d(x_context, filter_size[0], Fnum, 'contextCNN_convA{}'.format(layer), contextCNNs_reuse)
                contextCNN_convB = conv1d(x_context, filter_size[1], Fnum, 'contextCNN_convB{}'.format(layer), contextCNNs_reuse)
                x_context = tf.multiply(contextCNN_convA, tf.nn.sigmoid(contextCNN_convB), name='context_gating{}'.format(layer))
                if batch_norm:
                    x_context = tf.layers.batch_normalization(x_context)

            # (?, 1, 128) -> (?, 1, 64)
            contextCNN_pool = maxpool(x_context, 1, 2, 'contextCNN_pool', contextCNNs_reuse)
            concated = contextCNN_pool

        else:
            contextCNN_convA = x_context
            contextCNN_convB = x_context
            for layer, Fnum in enumerate(num_filters):
                contextCNN_convA = conv1d(contextCNN_convA, filter_size[0],
                                          Fnum, 'contextCNN_convA{}'.format(layer), contextCNNs_reuse)
                contextCNN_convB = conv1d(contextCNN_convB, filter_size[1],
                                          Fnum, 'contextCNN_convB{}'.format(layer), contextCNNs_reuse)
                if batch_norm:
                    contextCNN_convA = tf.layers.batch_normalization(contextCNN_convA)
                    contextCNN_convB = tf.layers.batch_normalization(contextCNN_convB)

            # (?, 1, 128) -> (?, 1, 64)
            contextCNN_poolA = maxpool(contextCNN_convA, 1, 2, 'contextCNN_poolA', contextCNNs_reuse)
            contextCNN_poolB = maxpool(contextCNN_convB, 1, 2, 'contextCNN_poolB', contextCNNs_reuse)
            concated = tf.concat([contextCNN_poolA, contextCNN_poolB], axis=-1)

        if is_first:
            contextCNNs_reuse = True
            is_first = False

        features = concated.shape[-1]
        contextCNNs[i] = tf.reshape(concated, [-1, features])

    features = contextCNNs[i].shape[-1]
    fc_reuse = False
    is_first = True
    # fc_outputs = []

    # Fully Connected Layer
    for i in range(max_sent):

        fc1_W = weight_variable([features, fc_hiddens], name='fc1_W', reuse=fc_reuse)
        fc1_b = bias_variable([fc_hiddens, ], name='fc1_b', reuse=fc_reuse)

        if batch_norm:
            fc1_out = tf.layers.batch_normalization(fc1_out)

        fc1_out = tf.nn.relu(tf.matmul(contextCNNs[i], fc1_W) + fc1_b)

        fc2_W = weight_variable([fc_hiddens, NDclasses], name='fc2_W', reuse=fc_reuse)
        fc2_b = bias_variable([NDclasses, ], name='fc2_b', reuse=fc_reuse)
        fc2_out = tf.matmul(fc1_out, fc2_W) + fc2_b

        # y_pre = fc2_out
        y_pre = tf.nn.softmax(fc2_out)

        y_pre = tf.expand_dims(y_pre, axis=1)

        if is_first:
            fc_outputs = y_pre
            fc_reuse = True
            is_first = False
        else:
            fc_outputs = tf.concat([fc_outputs, y_pre], axis=1)

    return fc_outputs
