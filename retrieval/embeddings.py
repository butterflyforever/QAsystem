from gensim.models import KeyedVectors
import numpy as np

class Embeddings(object):
    def __init__(self, emb_file):
        print('loading word vectors from '+ emb_file)
        print(emb_file)
        self.wv = KeyedVectors.load_word2vec_format(emb_file, binary=emb_file.endswith('.bin'))
        self.vocab = self.wv.vocab
        self.d_emb = self.wv.vector_size
    
    def __getitem__(self, key):
        try:
            return self.wv[key]
        except:
            if len(key) == 1:
                return None
            try:
                ret = np.zeros(self.d_emb)
                for char in key:
                    ret += self.wv[char]
                ret /= len(key)
                return ret
            except:
                return None