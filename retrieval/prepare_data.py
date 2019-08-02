from corpus import Corpus
from embeddings import Embeddings
from retriever import MaxMinWordVectorRetriever, TfidfWordVectorRetriever, AverageWordVectorRetriever
from generator import Generator
from configs import *
import numpy as np

if __name__ == '__main__':
    fquery = data_path + 'zhihu_query.txt'
    fresponse = data_path + 'zhihu_response.txt'
    ftest = data_path + 'turing.query'
    
    corpus_query = Corpus(fquery)
    corpus_response = Corpus(fresponse)

    emb = Embeddings(emb_path)

    retriever = TfidfWordVectorRetriever(corpus_query, corpus_response, emb, k=3)

    forigin = open(data_path + 'zhihu.query', 'w', encoding='utf8')
    fedit = open(data_path + 'zhihu.response', 'w', encoding='utf8')
    
    i = 0
    for query, response in zip(corpus_query, corpus_response):
        candidates, _ = retriever.retrieve([query.split()], query=False)
        candidates = candidates[0]
        if len(candidates):
            for candidate in candidates:
                if (not response in candidate) and len(candidate) > len(response):
                    score = np.dot(retriever.get_vector(response.split()), retriever.get_vector(candidate.split()))
                    if score > 0.8:
                        forigin.write(query + ' ： ' + response + '\n')
                        fedit.write(candidate + '\n')
                        if i % 100 == 0:
                            print('Query: ' + query)
                            print('Candidate: ' + candidate)
                            print('Response: ' + response)
        if i % 1000 == 0:
            print(i)
        i += 1

    forigin.close()
    fedit.close()

