fout = open('dataset.txt', 'w')
print('deal with dgk')
fin_ask = open('train_enc', 'r')
fin_ans = open('train_dec', 'r')
linecount = 0
for ask in fin_ask.readlines():
    ans = fin_ans.readline().strip()
    fout.write(ask.strip() + '\t' + ans + '\n')
    linecount += 1

fin_ask = open('test_enc', 'r')
fin_ans = open('test_dec', 'r')
for ask in fin_ask.readlines():
    ans = fin_ans.readline()
    fout.write(ask.strip() + '\t' + ans.strip() + '\n')
    linecount += 1

print(linecount)

print('deal with xiaohuangji')
fin = open('xiaohuangji50w_nofenci.conv', 'r')
mark = 1
ans = None
for line in fin.readlines():
    if line[0] == 'E':
        continue
    elif line[0] == 'M':
        mark = -mark
        if mark == -1:
            if ans is not None:
                # print(ask + '\t' + ans)
                fout.write(ask + '\t' + ans + '\n')
            ask = line[1:].strip()
        else:
            ans = line[1:].strip()
    else:
        if mark == -1:
            ask = ask + ', ' + line.strip()
        else:
            ans = ans + ', ' + line.strip()
fout.write(ask + '\t' + ans)
