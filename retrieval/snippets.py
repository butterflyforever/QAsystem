from .configs import *

f = open(corpus_path + 'zhihu.txt', encoding='utf8')
fq = open(data_path + 'zhihu_query.txt', 'w', encoding='utf8')
fr = open(data_path + 'zhihu_response.txt', 'w', encoding='utf8')

for line in f:
    query, response = line.strip().split('\t')
    fq.write(query + '\n')
    fr.write(response + '\n')

fq.close()
fr.close()

    