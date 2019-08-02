from corpus import Corpus
from embeddings import Embeddings
from retriever import MaxMinWordVectorRetriever, TfidfWordVectorRetriever, AverageWordVectorRetriever
from generator import Generator
from configs import *

import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = False


@app.route('/', methods=['POST'])
def api():
    try:
        print('start')
        queries = request.get_json()
        doc_split = []
        for query in queries:
            doc_split.append(query['content'])
        candidates, scores = retriever.retrieve(doc_split)
        ret = []
        for query, candidate, score in zip(queries, candidates, scores):
            if len(candidate):
                ret.append({
                    'userId': query['userId'],
                    'msgType': 'text',
                    'content': generator.generate(candidate),
                    'score': str(score[0])
                })
            else:
                ret.append({
                    'userId': query['userId'],
                    'msgType': 'none',
                    'content': ''
                })
        return jsonify(ret)
    except Exception as e:
        print(e)
        return jsonify([{"msgType": "none"}])


if __name__ == '__main__':
    fquery = data_path + 'corpus_query.txt'
    fresponse = data_path + 'corpus_response.txt'
    # ftest = corpus_path + 'turing.query'

    corpus_query = Corpus(fquery)
    corpus_response = Corpus(fresponse)

    emb = Embeddings(emb_path)

    # retriever = MaxMinWordVectorRetriever(corpus_query, corpus_response, emb)
    retriever = TfidfWordVectorRetriever(corpus_query, corpus_response, emb, k=1)
    # retriever = AverageWordVectorRetriever(corpus_query, corpus_response, emb)
    generator = Generator()

    # with open(ftest, encoding='utf8') as f:
    #    for line in f:
    #        candidates, _ = retriever.retrieve([line.strip().split(' ')], k=1, query=False)
    #        print('Query: %s\t\t Response:%s' % (line.strip(), ' '.join(generator.generate(candidates[0]))))

    app.run(host='0.0.0.0', port=8888)
