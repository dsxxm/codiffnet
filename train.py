from __future__ import division
import os, sys, shutil


from time import time
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #ignore the warnings 

from utils.Logging import Logging

def start(conf, data, model, evaluate):
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ----------------------
    # define log name 
    log_path = os.path.join(os.getcwd(), 'log/%s_%s.log' % (conf.data_name, conf.model_name))

    # ----------------------
    # start to initialize data for training and evaluating
    data.initializeRankingHandle()
    d_train, d_val, d_test, d_test_eva = data.train, data.val, data.test, data.test_eva
    
    print('System start to load data...')
    t0 = time()
    d_train.initializeRankingTrain()
    d_val.initializeRankingVT()
    d_test.initializeRankingVT()
    d_test_eva.initializeRankingEva()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    # ----------------------
    # prepare necessary data for diffnet++.
    print('System start to load graph...')
    data_dict = d_train.prepareModelSupplement(model)
    model.inputSupply(data_dict)
    model.startConstructGraph()

    # ----------------------
    # standard tensorflow running environment initialize
    tf_conf = tf.ConfigProto()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_conf)
    sess.run(model.init)

    if conf.pretrain_flag == 1:
        model.saver.restore(sess, conf.pre_model)
   
    log = Logging(log_path)
    print()
    log.record('Following will output the evaluation of the model:')

    # Start Training 
    for epoch in range(1, conf.epochs+1):

        tmp_train_loss = []
        tmp_base_loss = []
        tmp_user_cl_loss = []
        tmp_item_cl_loss = []
        t0 = time()

        #tmp_total_list = []
        while d_train.terminal_flag:

            d_train.getTrainRankingBatch()
            d_train.linkedMap()

            train_feed_dict = {}
            for (key, value) in model.map_dict['train'].items():
                train_feed_dict[key] = d_train.data_dict[value]

            [sub_train_loss, sub_base_loss, sub_user_cl, sub_item_cl, _] = sess.run(
                [model.map_dict['out']['train'],
                 model.map_dict['out']['train_base_loss'],
                 model.map_dict['out']['train_user_cl_loss'],
                 model.map_dict['out']['train_item_cl_loss'],
                 model.opt],
                feed_dict=train_feed_dict
            )
            tmp_train_loss.append(sub_train_loss)
            tmp_base_loss.append(sub_base_loss)
            tmp_user_cl_loss.append(sub_user_cl)
            tmp_item_cl_loss.append(sub_item_cl)


        train_loss = np.mean(tmp_train_loss)
        base_loss = np.mean(tmp_base_loss)
        user_cl = np.mean(tmp_user_cl_loss)
        item_cl = np.mean(tmp_item_cl_loss)
        t1 = time()

        # ----------------------
        # compute val loss and test loss
        d_val.getVTRankingOneBatch()
        d_val.linkedMap()
        val_feed_dict = {}
        for (key, value) in model.map_dict['val'].items():
            val_feed_dict[key] = d_val.data_dict[value]
        val_loss = sess.run(model.map_dict['out']['val'], feed_dict=val_feed_dict)


        d_test.getVTRankingOneBatch()
        d_test.linkedMap()
        test_feed_dict = {}
        for (key, value) in model.map_dict['test'].items():
            test_feed_dict[key] = d_test.data_dict[value]
        test_loss = sess.run(model.map_dict['out']['test'], feed_dict=test_feed_dict)
        t2 = time()

        # ----------------------
        # start evaluate model performance, hr and ndcg
        def getPositivePredictions():
            d_test_eva.getEvaPositiveBatch()
            d_test_eva.linkedRankingEvaMap()
            eva_feed_dict = {}
            for (key, value) in model.map_dict['eva'].items():
                eva_feed_dict[key] = d_test_eva.data_dict[value]
            positive_predictions = sess.run(
                model.map_dict['out']['eva'],
                feed_dict=eva_feed_dict
            )
            return positive_predictions

        def getNegativePredictions():
            negative_predictions = {}
            terminal_flag = 1
            while terminal_flag:
                batch_user_list, terminal_flag = d_test_eva.getEvaRankingBatch()
                d_test_eva.linkedRankingEvaMap()
                eva_feed_dict = {}
                for (key, value) in model.map_dict['eva'].items():
                    eva_feed_dict[key] = d_test_eva.data_dict[value]
                index = 0
                tmp_negative_predictions = np.reshape(
                    sess.run(
                        model.map_dict['out']['eva'],
                        feed_dict=eva_feed_dict
                    ),
                    [-1, conf.num_evaluate])
                for u in batch_user_list:
                    negative_predictions[u] = tmp_negative_predictions[index]
                    index = index + 1
            return negative_predictions

        tt2 = time()

        index_dict = d_test_eva.eva_index_dict
        positive_predictions = getPositivePredictions()
        negative_predictions = getNegativePredictions()
        
        # prepare for new batch
        d_test_eva.index = 0 

        hr_5, ndcg_5 = evaluate.evaluateRankingPerformance(\
            index_dict, positive_predictions, negative_predictions, conf.top5, conf.num_procs)
        hr_10, ndcg_10 = evaluate.evaluateRankingPerformance(\
            index_dict, positive_predictions, negative_predictions, conf.top10, conf.num_procs)
        hr_15, ndcg_15 = evaluate.evaluateRankingPerformance(\
            index_dict, positive_predictions, negative_predictions, conf.top15, conf.num_procs)
        
        tt3 = time()

        # ----------------------
        # print log to console and log_file
        log.record('Epoch:%d, cost:%.4fs, total:%.4f, base:%.4f, u_cl:%.4f, i_cl:%.4f, val:%.4f, test:%.4f' %
            (epoch, (t2-t0), train_loss, base_loss, user_cl, item_cl, val_loss, test_loss))
        log.record('Evaluate cost:%.4fs \n Top5: hr:%.4f, ndcg:%.4f \n Top10: hr:%.4f, ndcg:%.4f \n Top15: hr:%.4f, ndcg:%.4f' % ((tt3-tt2), hr_5, ndcg_5,hr_10, ndcg_10, hr_15, ndcg_15))

        d_train.generateTrainNegative()
