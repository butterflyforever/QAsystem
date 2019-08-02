import random

lines = []
with open("/data/share/corpus/corpus.txt",'r') as f:
	lines = f.readlines()

print(lines[:10])

random.shuffle(lines)

print(lines[:10])

with open("/data/share/corpus/new_corpus.txt",'w') as f:
	for l in lines:
		f.write(l)
