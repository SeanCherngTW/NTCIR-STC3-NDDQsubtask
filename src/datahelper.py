import re
import json
import param
import pickle
import logging
import operator
import unicodedata
import stctokenizer
import numpy as np
from pprint import pprint
from random import shuffle
from collections import Counter
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors

from gensim.models import word2vec
from gensim import models

from nltk.corpus import stopwords
sw = stopwords.words("english")


class DataHelper:
    def __init__(self, embedding_path):
        self.doclen = param.doclen
        self.embsize = param.embsize
        self.max_sent = param.max_sent
        self.NDclasses = param.NDclasses
        self.glove_300d = pickle.load(open('GloVe_840B_300d.p', 'rb'))

        with open('../data/test_data_en.json', encoding='utf8') as f:
            self.test_en = json.load(f)
        with open('../data/train_data_en.json', encoding='utf8') as f:
            self.train_en = json.load(f)
        self.stctokenizer = stctokenizer.STCTokenizer()

        self.train_corpus = pickle.load(open("train_corpus.p", "rb"))
        self.dev_corpus = pickle.load(open("dev_corpus.p", "rb"))
        self.test_corpus = self.get_raw_corpus('test')
        self.word_vectors = KeyedVectors.load(embedding_path)
        self.testIDs = [corpus[0] for corpus in self.test_corpus]

    def set_word_vectors(self, word_vectors):
        self.word_vectors = word_vectors
        self.embsize = word_vectors['a'].shape[0]

    def get_corpus(self, name):
        if name not in ['train', 'test']:
            raise ValueError('name must be train or test')
        if name == 'train':
            dataset = self.train_en
        else:
            dataset = self.test_en
        customer_corpus = []
        helpdesk_corpus = []
        for data in dataset:
            id = data['id']
            turns = data['turns']
            utts = []
            nuggets = []
            for tag in turns:
                utts.append(tag['utterances'])

            if name == 'train':
                for tag in data['annotations']:
                    nuggets.append(tag['nugget'])
                uttcounters = self._nugget_score(utts, nuggets)

            for idx, turn in enumerate(turns):
                utt = turn['utterances']
                sender = turn['sender']
                corpus_token = []
                if sender == 'customer':
                    if name == 'train':
                        customer_corpus.append((' '.join(utt), uttcounters[idx][1]))
                    else:
                        customer_corpus.append((' '.join(utt)))
                elif sender == 'helpdesk':
                    if name == 'train':
                        helpdesk_corpus.append((' '.join(utt), uttcounters[idx][1]))
                    else:
                        helpdesk_corpus.append((' '.join(utt)))
                else:
                    assert False, 'Sender name error'

        return customer_corpus, helpdesk_corpus

    def get_raw_corpus(self, name):
        if name not in ['train', 'test']:
            raise ValueError('name must be train or test')
        if name == 'train':
            dataset = self.train_en
        else:
            dataset = self.test_en
        customer_corpus = []
        helpdesk_corpus = []
        raw_corpus = []
        for data in dataset:
            id = data['id']
            turns = data['turns']
            utts = []
            nuggets = []
            quality = []
            for tag in turns:
                utts.append(tag['utterances'])

            if name == 'train':
                for tag in data['annotations']:
                    nuggets.append(tag['nugget'])
                    quality.append(tag['quality'])
                utt_nuggets = self._nugget_score(utts, nuggets)
                DQ = self._quality_score(utts, quality)

            turn_corpus = []
            turn_sender = []
            turn_nugget = []

            for idx, turn in enumerate(turns):
                utt = turn['utterances']
                sender = turn['sender']
                corpus_token = []
                if sender == 'customer':
                    if name == 'train':
                        turn_sender.append('customer')
                        turn_corpus.append(' '.join(utt))
                        turn_nugget.append(utt_nuggets[idx][1])
                    else:
                        turn_sender.append('customer')
                        turn_corpus.append(' '.join(utt))
                elif sender == 'helpdesk':
                    if name == 'train':
                        turn_sender.append('helpdesk')
                        turn_corpus.append(' '.join(utt))
                        turn_nugget.append(utt_nuggets[idx][1])
                    else:
                        turn_sender.append('helpdesk')
                        turn_corpus.append(' '.join(utt))
                else:
                    assert False, 'Sender name error'

            if name == 'train':
                raw_corpus.append((id, turn_sender, turn_corpus, turn_nugget, DQ))
            else:
                raw_corpus.append((id, turn_sender, turn_corpus))

        return raw_corpus

    def split_train_dev(self, train_corpus, ratio):
        shuffle(train_corpus)
        split_index = int(len(train_corpus) * ratio)
        train = train_corpus[:split_index]
        dev = train_corpus[split_index:]
        assert len(train) + len(dev) == len(train_corpus), 'Split error'
        return train, dev

    def _nugget_score(self, utts, nuggets):
        """ @brief 計算utterance的nugget分數，分數即為被標記為正確答案的次數
        @param utts 輸入的utterences: list
        @param nuggets utts對應的nuggets: list
        @return utts: list, counters: list, counters為對應至utts的nugget分數
        """
        uttcounters = []

        # 將nugget資料格式反轉 n x 19 -> 19 x n
        # 方便做後續的計算
        nuggets_T = list(map(list, zip(*nuggets)))
        for n, utt in zip(nuggets_T, utts):
            c = Counter(n)
            for key in c:
                c[key] /= 19
            uttcounters.append((utt, c))

        return uttcounters

    def _quality_score(self, utts, quality):
        """ @brief 計算utterance的quality分數，分數即為被標記為正確答案的次數
        @param utts 輸入的utterences: list
        @param quality utts對應的quality: list
        @return utts: list, counters: list, counters為對應至utts的quality分數
        """
        DQ = {}
        qualityA = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        qualityS = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        qualityE = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        quality_keys = [-2, -1, 0, 1, 2]

        for q in quality:
            qualityA[q['A']] += 1
            qualityS[q['S']] += 1
            qualityE[q['E']] += 1

        for k in quality_keys:
            qualityA[k] /= 19
            qualityS[k] /= 19
            qualityE[k] /= 19

        DQ['A'] = [qualityA[k] for k in sorted(qualityA.keys())]
        DQ['S'] = [qualityS[k] for k in sorted(qualityS.keys())]
        DQ['E'] = [qualityE[k] for k in sorted(qualityE.keys())]
        return DQ

    def prepare_word_embedding_corpus(self, wiki_path=None, token_type='nltk', remove_stopwords=False, to_lower=True):
        """
        @return corpus for word2vec training
        """
        corpus = []
        count = []
        logger = logging.getLogger('word embedding')
        if wiki_path:
            logger.info('Reading wiki text8 ...')
            with open(wiki_path, 'r') as f:
                text8 = self.stctokenizer.tokenize(token_type, f.readline(), remove_stopwords, to_lower)
                corpus.append(text8)

        logger.info('Tokenizing word embedding corpus ...')
        customer, helpdesk = self.get_corpus('train')
        for c in customer:
            text, _ = c
            tokens = self.stctokenizer.tokenize(token_type, text, remove_stopwords, to_lower)
            corpus.append(tokens)
            count += tokens

        for h in helpdesk:
            text, _ = h
            tokens = self.stctokenizer.tokenize(token_type, text, remove_stopwords, to_lower)
            corpus.append(tokens)
            count += tokens

        customer, helpdesk = self.get_corpus('test')
        for c in customer:
            text = c
            tokens = self.stctokenizer.tokenize(token_type, text, remove_stopwords, to_lower)
            corpus.append(tokens)
            count += tokens

        for h in helpdesk:
            text = h
            tokens = self.stctokenizer.tokenize(token_type, text, remove_stopwords, to_lower)
            corpus.append(tokens)
            count += tokens

        vocab = Counter(count)
        logger.info('STC-3 Data Vocabulary Count')
        logger.info('# All vocab        = {}'.format(sum(vocab.values())))
        logger.info('# Unique vocab     = {}'.format(len(vocab)))
        logger.info('# vocab (freq > 1) = {}'.format(len([i for i, v in vocab.items() if v > 1])))
        logger.info('# vocab (freq > 2) = {}'.format(len([i for i, v in vocab.items() if v > 2])))

        dl = 0
        doclens = []
        doclen_stat = [0] * 7
        for c in corpus[1:]:
            dl = max(dl, len(c))
            doclens.append(len(c))

        for d in doclens:
            if d <= 50:
                doclen_stat[0] += 1
            elif d <= 100:
                doclen_stat[1] += 1
            elif d <= 150:
                doclen_stat[2] += 1
            elif d <= 200:
                doclen_stat[3] += 1
            elif d <= 300:
                doclen_stat[4] += 1
            elif d <= 400:
                doclen_stat[5] += 1
            else:
                doclen_stat[6] += 1

        logger.info('max document len = {}'.format(dl))
        # print('document len distribution = {}'.format(doclen_stat))

        return corpus

    def test_w2v_model(self, word_vectors, word, topn):
        logger = logging.getLogger('word embedding')
        logger.info('Training word2vec model ...')
        res = word_vectors.most_similar(word, topn=topn)
        for item in res:
            print(item[0] + "," + str(item[1]))

    # def corpus_reshape(self, corpus):
    #     for i in range(len(corpus)):
    #         dialog, sentence, word = corpus[i].shape
    #         corpus[i] = np.reshape(corpus[i], (dialog * sentence, word))
    #     return corpus

    # def nd_reshape(self, nd_label):
    #     for i in range(len(nd_label)):
    #         dialog, l = nd_label[i].shape
    #         nd_label[i] = np.reshape(nd_label[i], (dialog * l))
    #     return nd_label

    def turn2mask(self, turns):
        # {'CNUG*': 0, 'CNUG': 1, 'CNaN': 2, 'CNUG0': 3, 'HNUG*': 4, 'HNUG': 5, 'HNaN': 6}
        all_dialog_masks = []
        for turn in turns:
            dialog_mask = []
            for i in range(self.max_sent):
                if i < turn:
                    if i % 2 == 0:  # customer
                        dialog_mask.append(np.concatenate((np.ones(4), np.zeros(3))))
                    else:  # helpdesk
                        dialog_mask.append(np.concatenate((np.zeros(4), np.ones(3))))
                else:
                    dialog_mask.append(np.zeros(self.max_sent))

            dialog_mask = np.asarray(dialog_mask)
            all_dialog_masks.append(np.asarray(dialog_mask.copy()))
        return all_dialog_masks

    def get_model_train_data(self, data_type, token_type, remove_stopwords, to_lower, emb, bert):
        logger = logging.getLogger('corpus word2vec')
        NDdim = 7
        DQdim = 3
        NDmap = {'CNUG*': 0, 'CNUG': 1, 'CNaN': 2, 'CNUG0': 3, 'HNUG*': 4, 'HNUG': 5, 'HNaN': 6}
        DQmap = {'A': 0, 'S': 1, 'E': 2}

        if bert:
            from bert_serving.client import BertClient
            bc = BertClient()

        # Store text, nuggets and quality for each dialog
        X = []
        Ynd = []
        Ydq = []
        turns = []
        bertX = []
        unk = 0

        if data_type == 'train':
            corpus = self.train_corpus
        elif data_type == 'dev':
            corpus = self.dev_corpus
        else:
            raise NameError('Parameter "data_type" must be "train" or "dev"')

        if emb == 'glove':
            embedding_index = self.glove_300d
            unk_vec = embedding_index['unk']
        elif emb == 'stc':
            embedding_index = self.word_vectors
        else:
            raise NameError('Parameter "emb" must be "glove" of "stc"')

        maxlen = 0

        for c in corpus:
            _id, _, texts, nuggets, quality = c

            # Store text, nuggets and quality for each dialog
            dialogX = []
            dialogbertX = []
            dialogND = []
            dialogDQ = []

            # For Nugget Detection
            for text, nugget in zip(texts, nuggets):
                text = '.' if text == '' else text
                tokens = self.stctokenizer.tokenize(token_type, text, remove_stopwords, to_lower)
                if bert:
                    vec = np.reshape(bc.encode([text]), 1024)
                    dialogbertX.append(vec)

                tokens_vec = []
                labelND = [0] * NDdim

                for token in tokens:
                    if len(tokens_vec) >= self.doclen:
                        break
                    if token in embedding_index:
                        tokens_vec.append(embedding_index[token])
                    else:
                        unk += 1
                        logger.info('Unknown word: {}'.format(token))
                        if emb == 'glove':
                            tokens_vec.append(unk_vec)
                        else:
                            tokens_vec.append(np.zeros(self.embsize))

                # doc補零
                while len(tokens_vec) < self.doclen:
                    tokens_vec.append(np.zeros(self.embsize))

                dialogX.append(np.asarray(tokens_vec.copy(), dtype=np.float32))

                maxlen = max(maxlen, len(tokens_vec))

                for k, v in nugget.items():
                    labelND[NDmap[k]] = nugget[k]

                dialogND.append(np.asarray(labelND.copy(), dtype=np.float32))

            # Num of turns for each dialog
            turns.append(len(dialogND))

            # Pending with zero for dialogs with turns < 7
            while len(dialogX) < self.max_sent and len(dialogND) < self.max_sent:
                dialogX.append(np.zeros([self.doclen, self.embsize]))
                dialogND.append(np.asarray([0] * self.NDclasses))
                if bert:
                    dialogbertX.append(np.zeros([1024, ]))

            X.append(np.asarray(dialogX.copy(), dtype=np.float32))
            Ynd.append(np.asarray(dialogND.copy(), dtype=np.float32))
            Ydq.append(quality.copy())
            if bert:
                bertX.append(np.asarray(dialogbertX))

        logger.info('Training data unknown words count: {}'.format(unk))
        logger.info('Training data max doclen: {}'.format(maxlen))
        masks = self.turn2mask(turns)
        return np.asarray(X), np.asarray(bertX), np.asarray(Ynd), np.asarray(Ydq), turns, masks

    def get_model_test_data(self, token_type, remove_stopwords, to_lower, emb, bert):
        logger = logging.getLogger('corpus word2vec')
        X = []
        turns = []
        bertX = []
        unk = 0

        if bert:
            from bert_serving.client import BertClient
            bc = BertClient()

        corpus = self.test_corpus
        maxlen = 0

        if emb == 'glove':
            embedding_index = self.glove_300d
            unk_vec = embedding_index['unk']
        elif emb == 'stc':
            embedding_index = self.word_vectors
        else:
            raise NameError('Parameter "emb" must be "glove" of "stc"')

        for c in corpus:
            _id, _, texts = c

            dialogX = []
            dialogbertX = []

            for text in texts:
                text = '.' if text == '' else text
                tokens = self.stctokenizer.tokenize(token_type, text, remove_stopwords, to_lower)
                if bert:
                    vec = np.reshape(bc.encode([text]), 1024)
                    dialogbertX.append(vec)
                tokens_vec = []

                for token in tokens:
                    if len(tokens_vec) >= self.doclen:
                        break
                    if token in embedding_index:
                        tokens_vec.append(embedding_index[token])
                    else:
                        unk += 1
                        logger.info('Unknown word: {}'.format(token))
                        if emb == 'glove':
                            tokens_vec.append(unk_vec)
                        else:
                            tokens_vec.append(np.zeros(self.embsize))

                # doc補零
                while len(tokens_vec) < self.doclen:
                    tokens_vec.append(np.zeros(self.embsize))

                dialogX.append(np.asarray(tokens_vec.copy(), dtype=np.float32))

                maxlen = max(maxlen, len(tokens_vec))

            # Num of turns for each dialog
            turns.append(len(dialogX))

            # Pending with zero for dialogs with turns < 7
            while len(dialogX) < self.max_sent:
                dialogX.append(np.zeros([self.doclen, self.embsize]))
                if bert:
                    dialogbertX.append(np.zeros([1024, ]))

            X.append(np.asarray(dialogX.copy(), dtype=np.float32))
            if bert:
                bertX.append(np.asarray(dialogbertX))
        logger.info('Testing data unknown words count: {}'.format(unk))
        logger.info('Testing data max doclen: {}'.format(maxlen))
        masks = self.turn2mask(turns)
        return np.asarray(X), np.asarray(bertX), turns, masks

    def pred_to_submission(self, testND, testDQA, testDQS, testDQE, turns, IDs, filename):
            # {'CNUG*': 0, 'CNUG': 1, 'CNaN': 2, 'CNUG0': 3, 'HNUG*': 4, 'HNUG': 5, 'HNaN': 6}
        assert len(testND) == len(testDQA) == len(testDQS) == len(testDQE) == len(turns)

        json.encoder.FLOAT_REPR = lambda x: format(x, '.10f')

        def json_float(f):
            return float(format(f, '.10f'))

        all_dialogs = []

        for nd, dqa, dqs, dqe, turn, _id in zip(testND, testDQA, testDQS, testDQE, turns, IDs):  # p is (49, )
            if nd.shape == (390, 7, 7):
                nd = nd.reshape(390, 49)
            sents_nuggets = np.array_split(nd, self.max_sent)
            dialog_json = {}
            quality_json = {}
            nugget_json = []

            # Dialog Quality
            assert len(dqa) == len(dqs) == len(dqe)
            quality_json['A'] = {'-2': json_float(dqa[0]),
                                 '-1': json_float(dqa[1]),
                                 '0': json_float(dqa[2]),
                                 '1': json_float(dqa[3]),
                                 '2': json_float(dqa[4]), }

            quality_json['S'] = {'-2': json_float(dqs[0]),
                                 '-1': json_float(dqs[1]),
                                 '0': json_float(dqs[2]),
                                 '1': json_float(dqs[3]),
                                 '2': json_float(dqs[4]), }

            quality_json['E'] = {'-2': json_float(dqe[0]),
                                 '-1': json_float(dqe[1]),
                                 '0': json_float(dqe[2]),
                                 '1': json_float(dqe[3]),
                                 '2': json_float(dqe[4]), }

            # Nugget Detection
            for i, sent_nugget in enumerate(sents_nuggets):  # for each sentence
                sent_nugget = sent_nugget.reshape(-1)

                if i == turn:
                    break

                if i % 2 == 0:  # customer
                    customer_nuggets = {}
                    customer_nuggets['CNUG*'] = json_float(sent_nugget[0])
                    customer_nuggets['CNUG'] = json_float(sent_nugget[1])
                    customer_nuggets['CNaN'] = json_float(sent_nugget[2])
                    customer_nuggets['CNUG0'] = json_float(sent_nugget[3])
                    nugget_json.append(customer_nuggets.copy())
                else:  # helpdesk
                    helpdesk_nuggets = {}
                    helpdesk_nuggets['HNUG*'] = json_float(sent_nugget[4])
                    helpdesk_nuggets['HNUG'] = json_float(sent_nugget[5])
                    helpdesk_nuggets['HNaN'] = json_float(sent_nugget[6])
                    nugget_json.append(helpdesk_nuggets.copy())

            # ID
            dialog_json['id'] = _id
            dialog_json['quality'] = quality_json.copy()
            dialog_json['nugget'] = nugget_json.copy()

            all_dialogs.append(dialog_json.copy())

        with open(filename, 'w') as f:
            json.dump(all_dialogs, f, indent=2, sort_keys=True)
