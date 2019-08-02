from gensim.models import TfidfModel
from gensim import similarities
import numpy as np
from os.path import isfile, basename
import tensorflow as tf
import time

from configs import *

class Retriever(object):
    def __init__(self, corpus_query, corpus_response, threshold=0.3):
        self.corpus_query = corpus_query
        self.corpus_response = corpus_response
        self.threshold = threshold

    def compute_score(self, doc_split):
        raise NotImplementedError

    def retrieve(self, doc_split, query=False, split=False):
        if self.gpu:
            print(3)
            scores, idx = self.compute_score(doc_split)
            print(4)
            ret = []
            score = []
            for i in range(len(doc_split)):
                _ret = []
                _score = []
                for j in range(self.k):
                    if scores[i][j] > self.threshold:
                        if not query:
                            if not split:
                                _ret.append(self.corpus_response[idx[i][j]].replace(' ', ''))
                            else:
                                _ret.append(self.corpus_response[idx[i][j]])
                        else:
                            if not split:
                                _ret.append(self.corpus_query[idx[i][j]].replace(' ', ''))
                            else:
                                _ret.append(self.corpus_query[idx[i][j]])
                        _score.append(scores[i][j])
                ret.append(_ret)
                score.append(_score)
            return ret, score
        else:
            scores = self.compute_score(doc_split)
            start = time.time()
            ret = []
            score = []
            for i in range(len(doc_split)):
                _ret = []
                _score = []
                for j in np.argsort(scores[i])[-self.k:][::-1]:
                    if scores[i][j] > self.threshold:
                        if not query:
                            _ret.append(self.corpus_response[j].replace(' ', ''))
                        else:
                            _ret.append(self.corpus_query[j].replace(' ', ''))
                        _score.append(scores[i][j])
                ret.append(_ret)
                score.append(_score)
            print('retrieved after computing score, %.4fs taken' % (time.time()-start))
            return ret, score

class TfidfWordVectorRetriever(Retriever):
    def __init__(self, corpus_query, corpus_response, emb, k=10, gpu=False):
        super().__init__(corpus_query, corpus_response)
        self.emb = emb
        self.gpu = True
        self.k = k

        print('building tfidf model...')
        tmp_tfidfmodel = tmp_path + basename(corpus_query.fpath) + '.tfidfmodel'
        if isfile(tmp_tfidfmodel):
            self.model = TfidfModel.load(tmp_tfidfmodel)
        else:
            self.model = TfidfModel(corpus_query.bow, dictionary=corpus_query.dict)
            self.model.save(tmp_tfidfmodel)

        print('building TfidfWordVector index...')
        index_file = tmp_path + basename(corpus_query.fpath) + '.tfidfwvindex.npy'
        if isfile(index_file):
            self.index = np.load(index_file)
            print('loaded from ' + index_file)
        else:
            self.index = np.zeros((len(corpus_query), self.emb.d_emb), dtype=np.float32)
            for i, query in enumerate(corpus_query):
                self.index[i] = self.get_vector(query.split())
                if i % 1000 == 0:
                    print('\r' + str(i), end='')
            self.index = self.index.T
            np.save(index_file, self.index)
            print('built TfidfWordVector index...')
        
        # Build Tensorflow Graph for fast matmul
        if self.gpu:
            self.t_index_init = tf.placeholder(dtype=tf.float32, shape=self.index.shape)
            self.t_index = tf.Variable(self.t_index_init)
            self.query = tf.placeholder(tf.float32, shape=(None, self.emb.d_emb))
            self.score = self.query @ self.t_index
            self.score_topk = tf.nn.top_k(self.score, k=self.k)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
            self.session.run(tf.global_variables_initializer(), feed_dict={self.t_index_init: self.index})

    def get_vector(self, doc_split, normalize=True):
        ret = np.zeros(self.emb.d_emb, dtype=np.float32)
        bow = self.corpus_query.dict.doc2bow(doc_split)
        maxv = 0
        for k, v in self.model[bow]:
            maxv = max(v, maxv)
            embedding = self.emb[self.corpus_query.dict[k]]
            if embedding is not None:
                ret += v * embedding
        
        # add words with 0 tfidf (as important as the most important word)
        for word in doc_split:
            if not word in self.corpus_query.dict.token2id:
                embedding = self.emb[word]
                if embedding is not None:
                    ret += maxv * embedding
                
        if normalize:
            if np.linalg.norm(ret) != 0:
                ret /= np.linalg.norm(ret)
        return ret

    def compute_score(self, doc_split):
        start = time.time()
        query = np.zeros((len(doc_split), self.emb.d_emb), dtype=np.float32)
        for i in range(len(doc_split)):
            query[i] = self.get_vector(doc_split[i])
        if self.gpu:
            score, idx = self.session.run(self.score_topk, feed_dict={self.query: query})
            print('compute %d scores, %.4fs taken' % (len(doc_split), time.time()-start))
            return score, idx
        else:
            score = query @ self.index
            print('compute %d scores, %.4fs taken' % (len(doc_split), time.time()-start))
            return score

    def __del__(self):
        if self.gpu:
            self.session.close()



####################### Notice ####################
# Classes below are not maintained for deployment # 
###################################################

class TfidfRetriever(Retriever):
    def __init__(self, corpus_query, corpus_response):
        super().__init__(corpus_query, corpus_response)
        print('building tfidf index...')
        self.model = TfidfModel(self.corpus_query.bow, dictionary=self.corpus_query.dict)
        self.index = similarities.SparseMatrixSimilarity(self.corpus_query.bow, len(self.corpus_query))

    def compute_score(self, doc_split):
        return self.index[self.model[self.corpus_query.doc2bow(doc_split)]]
        

class MaxMinWordVectorRetriever(Retriever):
    def __init__(self, corpus_query, corpus_response, emb):
        super().__init__(corpus_query, corpus_response)
        self.emb = emb
        print('building MaxMinWordVector index...')
        index_file = '../data/maxminwvindex.npy'
        if isfile(index_file):
            self.index = np.load(index_file)
            print('loaded from ' + index_file)
        else:    
            self.index = np.zeros((len(corpus_query), 2*self.emb.d_emb))
            for i, query in enumerate(corpus_query):
                self.index[i] = self.get_vector(query)
            np.save(index_file, self.index)
    
    def get_vector(self, doc_split, normalize=True):
        vmax = np.zeros(self.emb.d_emb)
        vmin = np.zeros(self.emb.d_emb)
        for word in doc_split:
            embedding = self.emb[word]
            if embedding is not None:
                vmax = np.max((vmax, embedding), axis=0)
                vmin = np.min((vmin, embedding), axis=0)
        ret = np.r_[vmax, vmin]
        if normalize:
            if np.linalg.norm(ret) != 0:
                ret /= np.linalg.norm(ret)
        return ret

    def compute_score(self, doc_split):
        return np.dot(self.index, self.get_vector(doc_split))


class AverageWordVectorRetriever(Retriever):
    def __init__(self, corpus_query, corpus_response, emb):
        super().__init__(corpus_query, corpus_response)
        self.emb = emb
        print('building avgwv index...')
        index_file = '../data/avgwvindex.npy'
        if isfile(index_file):
            self.index = np.load(index_file)
            print('loaded from ' + index_file)
        else:
            self.index = np.zeros((len(corpus_query), self.emb.d_emb))
            for i, query in enumerate(corpus_query):
                self.index[i] = self.get_vector(query)
            np.save(index_file, self.index)
    
    def get_vector(self, doc_split, normalize=True):
        ret = np.zeros(self.emb.d_emb)

        for word in doc_split:
            embedding = self.emb[word]
            if embedding is not None:
                ret += embedding
                
        if normalize:
            if np.linalg.norm(ret) != 0:
                ret /= np.linalg.norm(ret)
        return ret

    def compute_score(self, doc_split):
        return np.dot(self.index, self.get_vector(doc_split))

class ExpandTfidfRetriever(Retriever):
    def __init__(self, corpus_query, corpus_response, emb, k=10):
        super().__init__(corpus_query, corpus_response)
        self.emb = emb
        self.k = k
        print('building expandtfidf index...')
        self.model = TfidfModel(self.corpus_query.bow, dictionary=self.corpus_query.dict)
        self.index = similarities.SparseMatrixSimilarity(self.corpus_query.bow, len(self.corpus_query))

    def compute_score(self, doc_split):
        expanded = list()
        for word in doc_split:
            expanded.append(word)
            for s, _ in self.emb.wv.similar_by_vector(self.emb[word], topn=self.k+1)[1:]:
                expanded.append(s)
        return self.index[self.model[self.corpus_query.doc2bow(expanded)]]
