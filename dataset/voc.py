import jieba
from collections import defaultdict

fout = open('voc', 'w')
fin = open('dataset.txt', 'r')
dic = defaultdict(int)
START_VOCAB = ['<PAD>', '<GO>', '<EOS>', '<UNK>']
for line in fin.readlines():
    ask, ans = line.split('\t')
    words = jieba.lcut(ask.strip()) + jieba.lcut(ans.strip())
    for word in words:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1
vocabulary_list = START_VOCAB + sorted(dic, key=dic.get, reverse=True)
if len(vocabulary_list) > 50000:
    vocabulary_list = vocabulary_list[: 50000]
for voc in vocabulary_list:
    fout.write(voc + '\n')
