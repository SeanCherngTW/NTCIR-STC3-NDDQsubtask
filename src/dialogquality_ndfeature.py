import os
import param
import timeit
import logging
import datahelper
import stctokenizer
import tensorflow as tf
from collections import Counter
from gensim.models import Word2Vec
from gensim.models import word2vec
from time import gmtime, strftime, localtime
from gensim.models.keyedvectors import KeyedVectors


doclen = param.doclen
embsize = param.embsize
max_sent = param.max_sent
NDclasses = param.NDclasses
DQclasses = param.DQclasses
logger = logging.getLogger('DQ task')
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


def build_multistackCNN(x_split, bs, filter_size, num_filters, gating, batch_norm, nd):
    is_first = True
    sentCNNs_reuse = False
    nd = tf.unstack(nd, axis=1)

    with tf.name_scope('SentCNN'):
        for i, x_sent in enumerate(x_split):
            logger.debug('sentCNN input shape {}'.format(x_sent.shape))
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
                concated = tf.concat([sentCNN_pool, speaker, tf.expand_dims(nd[i], axis=1)], axis=-1, name='sentCNN_concated')

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

                sentCNN_poolA = maxpool(sentCNN_convA, doclen, 1, 'sentCNN_poolA', sentCNNs_reuse)
                sentCNN_poolB = maxpool(sentCNN_convB, doclen, 1, 'sentCNN_poolB', sentCNNs_reuse)
                concated = tf.concat([sentCNN_poolA, sentCNN_poolB, speaker, tf.expand_dims(nd[i], axis=1)], axis=-1, name='sentCNN_concated')

            logger.debug('sentCNN output shape {}'.format(concated.shape))

            if is_first:
                sentCNNs = concated
                sentCNNs_reuse = True
                is_first = False
            else:
                sentCNNs = tf.concat([sentCNNs, concated], axis=1, name='sentCNN_concate_sent')

        logger.debug('sentCNNs output shape {}'.format(sentCNNs.shape))
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


def build_FC(rnn_output, rnn_hiddens, batch_norm, type):
    logger.debug('FC Input {}'.format(str(rnn_output.shape)))  # (?, 7, 1024)
    if type == 'last':
        rnn_output = tf.unstack(rnn_output, axis=1)[-1]
    elif type == 'mean':
        rnn_output = tf.reduce_mean(rnn_output, axis=1)
    else:
        raise NameError('type must be "last" or "mean"')

    logger.debug('FC Input per sent {}'.format(str(rnn_output.shape)))
    with tf.name_scope('FCLayer'):
        if batch_norm:
            rnn_output = tf.layers.batch_normalization(rnn_output)

        fc1_W = weight_variable([rnn_hiddens, DQclasses], name='fc1_W')
        fc1_b = bias_variable([DQclasses, ], name='fc1_b')
        fc1_out = tf.matmul(rnn_output, fc1_W) + fc1_b
        y_pre = tf.nn.softmax(fc1_out)

    logger.debug('FC output y_pre {}'.format(str(y_pre.shape)))
    return y_pre


def init_input(doclen, embsize):
    # doclen = 150, embsize = 256
    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, max_sent, doclen, embsize], name='input_X')
        y = tf.placeholder(tf.float32, [None, DQclasses], name='output_Y')
        bs = tf.placeholder(tf.int32, [], name='batch_size')
        turns = tf.placeholder(tf.int32, [None, ], name='turns')
        num_dialog = tf.placeholder(tf.int32, [], name='num_dialog')
        nd = tf.placeholder(tf.float32, [None, max_sent, NDclasses], name='nd')
    return x, y, bs, turns, num_dialog, nd


def CNNRNN(x, bs, turns, keep_prob, rnn_hiddens, filter_size, num_filters, gating, batch_norm, num_layers, nd, memory_rnn_type=None):
    x_split = tf.unstack(x, axis=1)
    sentCNNs = build_multistackCNN(x_split, bs, filter_size, num_filters, gating, batch_norm, nd)  # Sentence representation
    logger.debug('sentCNNs input {}'.format(str(sentCNNs.shape)))
    rnn_output = build_RNN(sentCNNs, bs, turns, rnn_hiddens, batch_norm, 'context_RNN', 'Bi-LSTM', keep_prob=1, num_layers=num_layers)
    logger.debug('rnn_output input {}'.format(str(rnn_output.shape)))

    # Memory enhanced structure
    if memory_rnn_type:
        input_memory = build_RNN(rnn_output, bs, turns, rnn_hiddens, batch_norm, 'input_memory', memory_rnn_type, keep_prob=1, num_layers=1)
        output_memory = build_RNN(rnn_output, bs, turns, rnn_hiddens, batch_norm, 'output_memory', memory_rnn_type, keep_prob=1, num_layers=1)
        rnn_output = memory_enhanced(rnn_output, input_memory, output_memory)

    y_pre = build_FC(rnn_output, rnn_hiddens, batch_norm, 'last')

    return y_pre


def CNNCNN(x, bs, turns, keep_prob, fc_hiddens, filter_size, num_filters, gating, batch_norm, num_layers, nd, memory_rnn_type=None):
    x_split = tf.unstack(x, axis=1)

    sentCNNs = build_multistackCNN(x_split, bs, filter_size, num_filters, gating, batch_norm, nd)  # (?, 7, filter_num)
    sentCNNs = tf.unstack(sentCNNs, axis=1)  # (?, filter_num) * 7

    _contextCNNs = []
    contextCNNs_reuse = False
    is_first = True
    for i in range(max_sent):
        if i == 0:
            start = tf.fill((bs, sentCNNs[i].shape[-1]), 0.0)
            _contextCNNs.append(tf.concat([start, sentCNNs[i], sentCNNs[i + 1]], axis=-1))
        elif i == max_sent - 1:
            end = tf.fill((bs, sentCNNs[i].shape[-1]), 0.0)
            _contextCNNs.append(tf.concat([sentCNNs[i - 1], sentCNNs[i], end], axis=-1))
        else:
            _contextCNNs.append(tf.concat([sentCNNs[i - 1], sentCNNs[i], sentCNNs[i + 1]], axis=-1))

    # contextCNNs = (?, 3, filter_num) * 7

    for i in range(max_sent):
        logger.debug('_contextCNNs shape from {}'.format(_contextCNNs[i].shape))
        _, filters = _contextCNNs[i].shape
        _contextCNNs[i] = tf.reshape(_contextCNNs[i], [-1, 1, filters])
        logger.debug('_contextCNNs shape to {}'.format(_contextCNNs[i].shape))

    num_filters = [1024] * num_layers

    # Context CNNs
    for i, x_context in enumerate(_contextCNNs):
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
            contextCNNs = concated
            contextCNNs_reuse = True
            is_first = False
        else:
            contextCNNs = tf.concat([contextCNNs, concated], axis=1)
        # features = concated.shape[-1]
        # contextCNNs[i] = tf.reshape(concated, [-1, features])

    logger.debug('contextCNNs output shape {}'.format(str(contextCNNs.shape)))

    # memory_rnn_type = 'Bi-GRU'
    if memory_rnn_type:
        input_memory = build_RNN(contextCNNs, bs, turns, fc_hiddens, batch_norm, 'input_memory', memory_rnn_type, keep_prob=1, num_layers=1)
        output_memory = build_RNN(contextCNNs, bs, turns, fc_hiddens, batch_norm, 'output_memory', memory_rnn_type, keep_prob=1, num_layers=1)
        contextCNNs = memory_enhanced(contextCNNs, input_memory, output_memory)

    logger.debug('contextCNNs output shape {}'.format(str(contextCNNs.shape)))
    _, num_sent, num_features = contextCNNs.shape
    contextCNNs = tf.reshape(contextCNNs, [-1, num_sent * num_features])

    # contextCNNs = tf.reduce_mean(contextCNNs, axis=1)

    logger.debug('FC input shape {}'.format(str(contextCNNs.shape)))

    contextCNNs = tf.nn.dropout(contextCNNs, keep_prob)

    # Fully Connected Layer
    fc1_W = weight_variable([contextCNNs.shape[-1], fc_hiddens], name='fc1_W')
    fc1_b = bias_variable([fc_hiddens, ], name='fc1_b')
    fc1_out = tf.nn.relu(tf.matmul(contextCNNs, fc1_W) + fc1_b)

    if batch_norm:
        fc1_out = tf.layers.batch_normalization(fc1_out)

    fc2_W = weight_variable([fc_hiddens, DQclasses], name='fc2_W')
    fc2_b = bias_variable([DQclasses, ], name='fc2_b')
    fc2_out = tf.matmul(fc1_out, fc2_W) + fc2_b

    # y_pre = fc2_out
    y_pre = tf.nn.softmax(fc2_out)

    return y_pre
