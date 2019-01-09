import os
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
import nuggetdetection as ND
import dialogquality as DQ
import dialogquality_ndfeature as DQNDF
import stcevaluation as STCE
logger = logging.getLogger('Training')

doclen = param.doclen
embsize = param.embsize
max_sent = param.max_sent
NDclasses = param.NDclasses
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def start_trainND(
        trainX, trainY, train_turns, train_masks,
        devX, devND, dev_turns, dev_masks,
        testX, test_turns, test_masks,
        epoch, early_stopping, batch_size, lr, kp, hiddens, Fsize, Fnum, gating, bn, method, evaluate
):
    assert method.__name__ in ['CNNRNN', 'CNNCNN']

    tf.reset_default_graph()

    x, y, bs, turns, masks = ND.init_input(doclen, embsize)
    pred = method(x, bs, turns, kp, hiddens, Fsize, Fnum, gating, bn, masks)
    with tf.name_scope('loss'):
        cost = ND.loss_function(pred, y, batch_size, masks)
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    len_trainX = len(trainX)
    dev_batch = len(devX)

    method_info = '{}-stack{}'.format(len(Fnum), method.__name__)
    pred_test = None

    if gating:
        method_info = '{}-stack{}-gating'.format(len(Fnum), method.__name__)
    else:
        method_info = '{}-stack{}'.format(len(Fnum), method.__name__)
    tensorboard_path = "logs/{}/".format(method_info)
    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)

    min_JSD = 1
    min_RNSS = 1

    filter_size_str = '_'.join(list(map(str, Fsize)))
    num_filters_str = '_'.join(list(map(str, Fnum)))

    with tf.Session(config=config).as_default() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("logs/ND-{}/".format(method_info), sess.graph)
        for i in range(epoch):
            start = timeit.default_timer()
            merge = list(zip(trainX, trainY, train_turns, train_masks))
            random.shuffle(merge)
            trainX, trainY, train_turns, train_masks = zip(*merge)

            for start_idx in range(0, len_trainX, batch_size):
                end_idx = start_idx + batch_size
                if end_idx > len_trainX:
                    break
                batchX = trainX[start_idx:end_idx]
                batchY = trainY[start_idx:end_idx]
                batch_turns = train_turns[start_idx:end_idx]
                batch_masks = train_masks[start_idx:end_idx]
                train_bs = len(batchY)
                sess.run(train_op, feed_dict={x: batchX, y: batchY, bs: train_bs, turns: batch_turns, masks: batch_masks})

            pred_dev = sess.run(pred, feed_dict={x: devX, bs: dev_batch, turns: dev_turns, masks: dev_masks})
            RNSS, JSD = STCE.nugget_evaluation(pred_dev, devND, dev_turns)

            if JSD > min_JSD and RNSS > min_RNSS:
                current_early_stoping += 1
            else:
                current_early_stoping = 0
                min_JSD = JSD if JSD < min_JSD else min_JSD
                min_RNSS = RNSS if RNSS < min_RNSS else min_RNSS

            args = [method.__name__, i + 1, gating, bn, filter_size_str, hiddens,
                    num_filters_str, "{:.5f}".format(JSD), "{:.5f}".format(RNSS)]
            argstr = ','.join(map(str, args))
            logger.debug(argstr)

            if current_early_stoping >= early_stopping or (i + 1) == epoch:
                logger.info(argstr)
                break

        if evaluate:
            saver = tf.train.Saver()
            pred_test = sess.run(pred, feed_dict={x: testX, bs: len(testX), turns: test_turns, masks: test_masks})
            modelname = [method.__name__, i + 1, len(Fnum), batch_size, gating, filter_size_str, hiddens, num_filters_str]
            modelpath = 'models/ND/{}.ckpt'.format('-'.join(map(str, modelname)))
            saver.save(sess, modelpath)
            print('{} is saved\n'.format(modelpath))
            return pred_test


def start_trainDQ(
    trainX, trainY, train_turns,
    devX, devDQ, dev_turns, testX, test_turns,
    scoretype, epoch, early_stopping, batch_size, lr, kp, hiddens, Fsize,
    Fnum, gating, bn, method, evaluate
):

    assert method.__name__ in ['CNNRNN', 'CNNCNN']
    assert scoretype in ['DQA', 'DQS', 'DQE']

    tf.reset_default_graph()
    eps = 1e-10

    x, y, bs, turns = DQ.init_input(doclen, embsize)
    pred = method(x, bs, turns, kp, hiddens, Fsize, Fnum, gating, bn)
    with tf.name_scope('loss'):
        # dont need softmax in NN output
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0))))
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    len_trainX = len(trainX)
    dev_batch = len(devX)

    total_time = 0.0
    pred_test = None

    if gating:
        method_info = '{}-stack{}-gating'.format(len(Fnum), method.__name__)
    else:
        method_info = '{}-stack{}'.format(len(Fnum), method.__name__)
    tensorboard_path = "logs/DQ-{}/".format(method_info)
    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)

    min_NMD = 1
    min_RSNOD = 1

    filter_size_str = '_'.join(list(map(str, Fsize)))
    num_filters_str = '_'.join(list(map(str, Fnum)))

    with tf.Session(config=config).as_default() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        for i in range(epoch):
            merge = list(zip(trainX, trainY, train_turns))
            random.shuffle(merge)
            trainX, trainY, train_turns = zip(*merge)

            for start_idx in range(0, len_trainX, batch_size):
                end_idx = start_idx + batch_size
                if end_idx > len_trainX:
                    break
                batchX = trainX[start_idx:end_idx]
                batchY = trainY[start_idx:end_idx]
                batch_turns = train_turns[start_idx:end_idx]
                train_bs = len(batchY)
                sess.run(train_op, feed_dict={x: batchX, y: batchY, bs: train_bs, turns: batch_turns, })

            pred_dev = sess.run(pred, feed_dict={x: devX, bs: dev_batch, turns: dev_turns, })
            NMD, RSNOD = STCE.quality_evaluation(pred_dev, devDQ)

            if NMD > min_NMD and RSNOD > min_RSNOD:
                current_early_stoping += 1
            else:
                current_early_stoping = 0
                min_NMD = NMD if NMD < min_NMD else min_NMD
                min_RSNOD = RSNOD if RSNOD < min_RSNOD else min_RSNOD

            args = [method.__name__, i + 1, gating, bn, filter_size_str, hiddens,
                    num_filters_str, "{:.5f}".format(NMD), "{:.5f}".format(RSNOD)]
            argstr = ','.join(map(str, args))
            logger.debug(argstr)

            if current_early_stoping >= early_stopping or (i + 1) == epoch:
                logger.info(argstr)
                break

        if evaluate:
            saver = tf.train.Saver()
            pred_test = sess.run(pred, feed_dict={x: testX, bs: len(testX), turns: test_turns})
            modelname = [method.__name__, i + 1, len(Fnum), batch_size, gating, bn, filter_size_str, hiddens, num_filters_str]
            modelpath = 'models/{}/{}.ckpt'.format(scoretype, '-'.join(map(str, modelname)))
            saver.save(sess, modelpath)
            print('{} is saved\n'.format(modelpath))
            return pred_test


def start_trainDQ_NDF(
    trainX, trainY, train_turns, trainND,
    devX, devDQ, dev_turns, devND,
    testX, test_turns, testND,
    scoretype, epoch, early_stopping, batch_size, lr, kp, hiddens, Fsize,
    Fnum, gating, bn, method, evaluate
):

    assert method.__name__ in ['CNNRNN', 'CNNCNN']
    assert scoretype in ['DQA', 'DQS', 'DQE']

    tf.reset_default_graph()
    eps = 1e-10

    x, y, bs, turns, nd = DQNDF.init_input(doclen, embsize)
    pred = method(x, bs, turns, nd, kp, hiddens, Fsize, Fnum, gating, bn)

    with tf.name_scope('loss'):
         # dont need softmax in NN output
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0))))
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    len_trainX = len(trainX)
    dev_batch = len(devX)

    pred_test = None

    if gating:
        method_info = '{}-stack{}-gating'.format(len(Fnum), method.__name__)
    else:
        method_info = '{}-stack{}'.format(len(Fnum), method.__name__)

    tensorboard_path = "logs/{}/".format(method_info)
    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)

    min_NMD = 1
    min_RSNOD = 1

    filter_size_str = '_'.join(list(map(str, Fsize)))
    num_filters_str = '_'.join(list(map(str, Fnum)))

    with tf.Session(config=config).as_default() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        for i in range(epoch):
            merge = list(zip(trainX, trainY, train_turns))
            random.shuffle(merge)
            trainX, trainY, train_turns = zip(*merge)

            for start_idx in range(0, len_trainX, batch_size):
                end_idx = start_idx + batch_size
                if end_idx > len_trainX:
                    break
                batchX = trainX[start_idx:end_idx]
                batchY = trainY[start_idx:end_idx]
                batch_turns = train_turns[start_idx:end_idx]
                batch_nd = trainND[start_idx:end_idx]

                train_bs = len(batchY)
                sess.run(train_op, feed_dict={x: batchX, y: batchY, bs: train_bs, turns: batch_turns, nd: batch_nd, })

            pred_dev = sess.run(pred, feed_dict={x: devX, bs: dev_batch, turns: dev_turns, nd: devND, })
            NMD, RSNOD = STCE.quality_evaluation(pred_dev, devDQ)

            if NMD > min_NMD and RSNOD > min_RSNOD:
                current_early_stoping += 1
            else:
                current_early_stoping = 0
                min_NMD = NMD if NMD < min_NMD else min_NMD
                min_RSNOD = RSNOD if RSNOD < min_RSNOD else min_RSNOD

            args = [method.__name__, i + 1, gating, bn, filter_size_str, hiddens,
                    num_filters_str, "{:.5f}".format(NMD), "{:.5f}".format(RSNOD)]
            argstr = ','.join(map(str, args))
            logger.debug(argstr)

            if current_early_stoping >= early_stopping or (i + 1) == epoch:
                logger.info(argstr)
                break

        if evaluate:
            saver = tf.train.Saver()
            pred_test = sess.run(pred, feed_dict={x: testX, bs: len(testX), turns: test_turns, nd: testND, })
            modelname = [method.__name__, i + 1, len(Fnum), batch_size, gating, bn, filter_size_str, hiddens, num_filters_str]
            modelpath = 'models/{}/{}.ckpt'.format(scoretype, '-'.join(map(str, modelname)))
            saver.save(sess, modelpath)
            print('{} is saved\n'.format(modelpath))
            return pred_test
