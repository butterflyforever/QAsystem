from gensim.corpora import Dictionary
from gensim.corpora import TextCorpus, MmCorpus
from configs import *
from os.path import basename, isfile

class Corpus(object):
    def __init__(self, fpath):
        self.fpath = fpath
        with open(self.fpath, encoding='utf8') as f:
            self._corpus = [line.strip() for line in f.readlines()]
        print('Generating Dictionary and Bow for ' + fpath)
        tmp_mm = tmp_path + basename(fpath) + '.mm'
        tmp_dict = tmp_path + basename(fpath) + '.dict'
        if isfile(tmp_mm) and isfile(tmp_dict):
            self.dict = Dictionary.load(tmp_dict)
            self.bow = MmCorpus(tmp_mm)
        else:
            tmp_corpus = TextCorpus(self.fpath, token_filters=[(lambda x: x)], character_filters=[(lambda x: str(x, 'utf8'))])
            tmp_corpus.dictionary.save(tmp_dict)
            MmCorpus.serialize(tmp_mm, tmp_corpus)
            self.dict = tmp_corpus.dictionary
            self.bow = MmCorpus(tmp_mm)
            print('Generated')
        
    def __len__(self):
        return len(self._corpus)
    
    def __getitem__(self, i):
        return self._corpus[i]

    def __iter__(self):
        for line in self._corpus:
            yield line
    
    def doc2bow(self, doc):
        return self.dict.doc2bow(doc)