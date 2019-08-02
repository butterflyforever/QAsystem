fq = open('../../data/turing.query', encoding='utf8')
fr = open('../../data/turing.response', encoding='utf8')
fo = open('../../data/turing.pair', 'w', encoding='utf8')

for lineq in fq.readlines():
    liner = fr.readline()
    fo.write('Ask: ' + lineq.strip() + '\t\t' + 'Answer: ' + liner)

fo.close()
fq.close()
fr.close()