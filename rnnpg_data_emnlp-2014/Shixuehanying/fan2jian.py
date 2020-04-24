#!/usr/bin/python

fan2jian_dict = {}
fj_encode = 'utf-8'
fj_file = 'FJ_std_3.1k'

def load(infile):
    cnt = 0
    for line in open(infile):
        cnt += 1
        items = line.decode(fj_encode).strip().split()
        if len(items) != 2:
            print('not 2 characters, %d:%s' % (cnt, line))
            continue
        fan2jian_dict[items[0]] = items[1]

def fan2jian_char(fan):
    if len(fan2jian_dict) == 0:
        load(fj_file)
    jian = fan2jian_dict.get(fan)
    return jian if jian != None else fan


def fan2jian(fan):
    return ''.join([fan2jian_char(ch) for ch in fan])

def fan2jian4file(fanFile, jianFile, fanFEncode = 'utf-8', jianFEncode = 'utf-8'):
    fout = open(jianFile, 'w')
    for line in open(fanFile):
        utext = line.decode(fanFEncode)
        jian = fan2jian(utext)
        text = jian.encode(jianFEncode)
        fout.write(text)

    fout.close()
