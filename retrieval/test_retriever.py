from corpus import Corpus
from embeddings import Embeddings
from configs import *
from retriever import TfidfRetriever, MaxMinWordVectorRetriever, TfidfWordVectorRetriever, AverageWordVectorRetriever, \
    ExpandTfidfRetriever

if __name__ == '__main__':
    fquery = data_path + 'corpus_query.txt'
    fresponse = data_path + 'corpus_response.txt'

    corpus_query = Corpus(fquery)
    corpus_response = Corpus(fresponse)

    emb = Embeddings('./data/emb/sgns.zhihu.bigram-char')  # .bin')

    # retriever = TfidfRetriever(corpus_query, corpus_response)
    # retriever = MaxMinWordVectorRetriever(corpus_query, corpus_response, emb)
    retriever = TfidfWordVectorRetriever(corpus_query, corpus_response, emb, k=1)
    # retriever = AverageWordVectorRetriever(corpus_query, corpus_response, emb)
    # retriever = ExpandTfidfRetriever(corpus_query, corpus_response, emb, k=10)

    # with open(ftest, encoding='utf8') as f:
    #     for line in f:
    #         candidates, scores = retriever.retrieve(line.strip().split(' '), k=30, query=True)
    #         print('Origin: ' + line.strip())
    #         print('Retrieved: ')
    #         for i, candidate in enumerate(candidates):
    #             print('\t' + ' '.join(candidate) + ' ' + str(scores[i]))
    #         print('----------------------------------')

    line = "不不呵呵么"
    line = [line.strip().split(' ')]
    print(line)
    candidates, scores = retriever.retrieve(line, query=True)
    print(line)
    print('Retrieved: ')
    for i, candidate in enumerate(candidates):
        print('\t' + ' '.join(candidate) + ' ' + str(scores[i]))
    print('----------------------------------')
