import os
import sys
import time
import json
import pickle
import timeit
import random
import param
import shutil
import logging
import collections
import numpy as np
import tensorflow as tf
import datahelper
import stctokenizer
import stcevaluation as STCE
logger = logging.getLogger('Training')

doclen = param.doclen
embsize = param.embsize
max_sent = param.max_sent
NDclasses = param.NDclasses
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def start_trainND(
        trainX, trainY, train_turns, train_masks,
        devX, devND, dev_turns, dev_masks,
        testX, test_turns, test_masks,
        epoch, early_stopping, batch_size, lr, kp, hiddens, Fsize, Fnum, gating, bn, num_layers, method, evaluate, memory_rnn_type=None, bert=False,
):
    assert method.__name__ in ['CNNRNN', 'CNNCNN']

    if bert:
        import nuggetdetectionBERT as ND
    else:
        import nuggetdetection as ND

    tf.reset_default_graph()
    x, y, bs, turns, masks, num_sent = ND.init_input(doclen, embsize)

    pred = method(x, y, bs, turns, kp, hiddens, Fsize, Fnum, gating, bn, num_layers, masks, memory_rnn_type)
    with tf.name_scope('loss'):
        cost = ND.loss_function(pred, y, batch_size, num_sent, masks)

    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    len_train = len(trainX)
    len_dev = len(devX)

    method_info = '{}-stack{}'.format(len(Fnum), method.__name__)
    pred_test = None

    if gating:
        method_info = '{}-stack{}-gating'.format(len(Fnum), method.__name__)
    else:
        method_info = '{}-stack{}'.format(len(Fnum), method.__name__)
    tensorboard_path = "logs/{}/".format(method_info)
    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)

    min_dev_loss = sys.maxsize
    train_losses = []
    dev_losses = []

    train_num_sent = sum(train_turns)
    dev_num_sent = sum(dev_turns)

    filter_size_str = '_'.join(list(map(str, Fsize)))
    num_filters_str = '_'.join(list(map(str, Fnum)))

    with tf.Session(config=config).as_default() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("logs/ND-{}/".format(method_info), sess.graph)
        for e in range(epoch):
            merge = list(zip(trainX, trainY, train_turns, train_masks))
            random.shuffle(merge)
            trainX, trainY, train_turns, train_masks = zip(*merge)
            del merge

            for i in range(0, len_train, batch_size):
                j = i + batch_size
                train_bs = batch_size if j < len_train else len_train - i
                sess.run(train_op, feed_dict={x: trainX[i:j], y: trainY[i:j], bs: train_bs,
                                              turns: train_turns[i:j], masks: train_masks[i:j], num_sent: train_num_sent, })

            # Compute train loss in batch since memory is not enough
            train_loss = 0
            for i in range(0, len_train, batch_size):
                j = i + batch_size
                train_bs = len(trainY[i:j])
                train_loss += sess.run(cost, feed_dict={x: trainX[i:j], y: trainY[i:j], bs: train_bs,
                                                        turns: train_turns[i:j], masks: train_masks[i:j], num_sent: train_num_sent, })

            # Compute dev loss in batch since memory is not enough
            dev_loss = 0
            for i in range(0, len_dev, batch_size):
                j = i + batch_size
                dev_bs = len(devND[i:j])
                dev_loss += sess.run(cost, feed_dict={x: devX[i:j], y: devND[i:j], bs: dev_bs,
                                                      turns: dev_turns[i:j], masks: dev_masks[i:j], num_sent: dev_num_sent})

            print('epoch: {} | trainloss: {:.4f} | devloss: {:.4f}'.format(e + 1, train_loss, dev_loss))
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

            if dev_loss >= min_dev_loss:
                current_early_stoping += 1
            else:
                current_early_stoping = 0
                min_dev_loss = dev_loss
                saver = tf.train.Saver(tf.trainable_variables())
                saver.save(sess, './tmp/best_params')

            if current_early_stoping >= early_stopping or (e + 1) == epoch:
                # assert dev_losses.index(min(dev_losses)) == len(dev_losses) - early_stopping - 1, 'Early Stop Error'
                saver.restore(sess, './tmp/best_params')
                pred_dev = sess.run(pred, feed_dict={x: devX, bs: len_dev, turns: dev_turns, masks: dev_masks})

                RNSS, JSD = STCE.nugget_evaluation(pred_dev, devND, dev_turns, dev_masks)
                args = [method.__name__, e + 1, gating, bn, filter_size_str, hiddens, num_filters_str, kp, "{:.4f}".format(JSD), "{:.4f}".format(RNSS)]
                argstr = '|'.join(map(str, args))
                print(argstr)
                break

        if evaluate:
            pred_test = sess.run(pred, feed_dict={x: testX, bs: len(testX), turns: test_turns, masks: test_masks})
            # saver = tf.train.Saver()
            # modelname = [method.__name__, e + 1, len(Fnum), batch_size, gating, filter_size_str, hiddens, num_filters_str]
            # modelpath = 'models/ND/{}.ckpt'.format('-'.join(map(str, modelname)))
            # saver.save(sess, modelpath)
            # print('{} is saved\n'.format(modelpath))

        return pred_test, train_losses, dev_losses


def start_trainDQ(
    trainX, trainY, train_turns,
    devX, devY, dev_turns,
    testX, test_turns,
    scoretype, epoch, early_stopping, batch_size, lr, kp, hiddens, Fsize,
    Fnum, gating, bn, num_layers, method, evaluate, memory_rnn_type=None, bert=False,
):

    assert method.__name__ in ['CNNRNN', 'CNNCNN']
    assert scoretype in ['DQA', 'DQS', 'DQE']

    if bert:
        import dialogqualityBERT as DQ
    else:
        import dialogquality as DQ

    tf.reset_default_graph()

    x, y, bs, turns, num_dialog = DQ.init_input(doclen, embsize)
    pred = method(x, bs, turns, kp, hiddens, Fsize, Fnum, gating, bn, num_layers, memory_rnn_type)
    with tf.name_scope('loss'):
        cost = tf.divide(-tf.reduce_sum(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0))), tf.cast(num_dialog, tf.float32))
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    pred_test = None

    if gating:
        method_info = '{}-stack{}-gating'.format(len(Fnum), method.__name__)
    else:
        method_info = '{}-stack{}'.format(len(Fnum), method.__name__)
    tensorboard_path = "logs/DQ-{}/".format(method_info)
    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)

    min_dev_loss = sys.maxsize
    train_losses = []
    dev_losses = []

    len_train = len(trainX)
    len_dev = len(devX)

    filter_size_str = '_'.join(list(map(str, Fsize)))
    num_filters_str = '_'.join(list(map(str, Fnum)))

    with tf.Session(config=config).as_default() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        for e in range(epoch):
            merge = list(zip(trainX, trainY, train_turns))
            random.shuffle(merge)
            trainX, trainY, train_turns = zip(*merge)
            del merge

            for i in range(0, len_train, batch_size):
                j = i + batch_size
                train_bs = batch_size if j < len_train else len_train - i
                sess.run(train_op, feed_dict={x: trainX[i:j], y: trainY[i:j], bs: train_bs,
                                              turns: train_turns[i:j], num_dialog: len_train})

            # Compute train loss in batch since memory is not enough
            train_loss = 0
            for i in range(0, len_train, batch_size):
                j = i + batch_size
                train_bs = batch_size if j < len_train else len_train - i
                train_loss += sess.run(cost, feed_dict={x: trainX[i:j], y: trainY[i:j], bs: train_bs, turns: train_turns[i:j], num_dialog: len_train})

            # Compute dev loss in batch since memory is not enough
            dev_loss = 0
            for i in range(0, len_dev, batch_size):
                j = i + batch_size
                dev_bs = batch_size if j < len_dev else len_dev - i
                dev_loss += sess.run(cost, feed_dict={x: devX[i:j], y: devY[i:j], bs: dev_bs, turns: dev_turns[i:j], num_dialog: len_dev})

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

            if dev_loss >= min_dev_loss:
                current_early_stoping += 1
            else:
                current_early_stoping = 0
                min_dev_loss = dev_loss
                saver = tf.train.Saver(tf.trainable_variables())
                saver.save(sess, './tmp/best_params')

            if current_early_stoping >= early_stopping or (e + 1) == epoch:
                assert dev_losses.index(min(dev_losses)) == len(dev_losses) - early_stopping - 1, 'Early Stop Error'
                saver.restore(sess, './tmp/best_params')
                pred_dev = sess.run(pred, feed_dict={x: devX, bs: len_dev, turns: dev_turns, })
                NMD, RSNOD = STCE.quality_evaluation(pred_dev, devY)
                args = [method.__name__, e + 1, gating, bn, filter_size_str, hiddens,
                        num_filters_str, "{:.5f}".format(NMD), "{:.5f}".format(RSNOD)]
                argstr = '|'.join(map(str, args))
                print(argstr)
                break

        if evaluate:
            pred_test = sess.run(pred, feed_dict={x: testX, bs: len(testX), turns: test_turns})
            # saver = tf.train.Saver()
            # modelname = [method.__name__, e + 1, len(Fnum), batch_size, gating, filter_size_str, hiddens, num_filters_str]
            # modelpath = 'models/ND/{}.ckpt'.format('-'.join(map(str, modelname)))
            # saver.save(sess, modelpath)
            # print('{} is saved\n'.format(modelpath))

        return pred_test, train_losses, dev_losses


def start_trainDQ_NDF(
    trainX, trainY, train_turns, trainND,
    devX, devY, dev_turns, devND,
    testX, test_turns, testND,
    scoretype, epoch, early_stopping, batch_size, lr, kp, hiddens, Fsize,
    Fnum, gating, bn, num_layers, method, evaluate, memory_rnn_type=None, bert=False,
):

    assert method.__name__ in ['CNNRNN', 'CNNCNN']
    assert scoretype in ['DQA', 'DQS', 'DQE']

    if bert:
        import dialogquality_ndfeatureBERT as DQNDF
    else:
        import dialogquality_ndfeature as DQNDF

    tf.reset_default_graph()

    x, y, bs, turns, num_dialog, nd = DQNDF.init_input(doclen, embsize)
    pred = method(x, bs, turns, kp, hiddens, Fsize, Fnum, gating, bn, num_layers, nd, memory_rnn_type)
    with tf.name_scope('loss'):
        cost = tf.divide(-tf.reduce_sum(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0))), tf.cast(num_dialog, tf.float32))
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    pred_test = None

    if gating:
        method_info = '{}-stack{}-gating'.format(len(Fnum), method.__name__)
    else:
        method_info = '{}-stack{}'.format(len(Fnum), method.__name__)
    tensorboard_path = "logs/DQ-{}/".format(method_info)
    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)

    min_dev_loss = sys.maxsize
    train_losses = []
    dev_losses = []

    len_train = len(trainX)
    len_dev = len(devX)

    filter_size_str = '_'.join(list(map(str, Fsize)))
    num_filters_str = '_'.join(list(map(str, Fnum)))

    with tf.Session(config=config).as_default() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        for e in range(epoch):
            merge = list(zip(trainX, trainY, train_turns))
            random.shuffle(merge)
            trainX, trainY, train_turns = zip(*merge)
            del merge

            for i in range(0, len_train, batch_size):
                j = i + batch_size
                train_bs = batch_size if j < len_train else len_train - i
                sess.run(train_op, feed_dict={x: trainX[i:j], y: trainY[i:j], bs: train_bs,
                                              turns: train_turns[i:j], num_dialog: len_train, nd: trainND[i:j]})

            # Compute train loss in batch since memory is not enough
            train_loss = 0
            for i in range(0, len_train, batch_size):
                j = i + batch_size
                train_bs = batch_size if j < len_train else len_train - i
                train_loss += sess.run(cost, feed_dict={x: trainX[i:j], y: trainY[i:j], bs: train_bs, turns: train_turns[i:j], num_dialog: len_train, nd: trainND[i:j]})

            # Compute dev loss in batch since memory is not enough
            dev_loss = 0
            for i in range(0, len_dev, batch_size):
                j = i + batch_size
                dev_bs = batch_size if j < len_dev else len_dev - i
                dev_loss += sess.run(cost, feed_dict={x: devX[i:j], y: devY[i:j], bs: dev_bs, turns: dev_turns[i:j], num_dialog: len_dev, nd: devND[i:j]})

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

            if dev_loss >= min_dev_loss:
                current_early_stoping += 1
            else:
                current_early_stoping = 0
                min_dev_loss = dev_loss
                saver = tf.train.Saver(tf.trainable_variables())
                saver.save(sess, './tmp/best_params')

            if current_early_stoping >= early_stopping or (e + 1) == epoch:
                assert dev_losses.index(min(dev_losses)) == len(dev_losses) - early_stopping - 1, 'Early Stop Error'
                saver.restore(sess, './tmp/best_params')
                pred_dev = sess.run(pred, feed_dict={x: devX, bs: len_dev, turns: dev_turns, nd: devND})
                NMD, RSNOD = STCE.quality_evaluation(pred_dev, devY)
                args = [method.__name__, e + 1, gating, bn, filter_size_str, hiddens,
                        num_filters_str, "{:.5f}".format(NMD), "{:.5f}".format(RSNOD)]
                argstr = '|'.join(map(str, args))
                print(argstr)
                break

        if evaluate:
            pred_test = sess.run(pred, feed_dict={x: testX, bs: len(testX), turns: test_turns, nd: testND})
            # saver = tf.train.Saver()
            # modelname = [method.__name__, e + 1, len(Fnum), batch_size, gating, filter_size_str, hiddens, num_filters_str]
            # modelpath = 'models/ND/{}.ckpt'.format('-'.join(map(str, modelname)))
            # saver.save(sess, modelpath)
            # print('{} is saved\n'.format(modelpath))

        return pred_test, train_losses, dev_losses
