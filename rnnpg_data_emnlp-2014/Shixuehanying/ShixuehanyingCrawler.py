#!/usr/bin/python
#-*-coding:UTF-8-*-

import urllib
import urllib3
import re, sys, os, time
from fan2jian import *
import argparse

def POST(url, values, outfile):
    data = urllib.urlencode(values)
    print(data)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    page = response.read()

    fout = open(outfile, 'w')
    fout.write(page)
    fout.close()

    print('%s done!' % url)

def getKeywordsIDs(aspPage):
    page = open(aspPage).read().decode('big5')
    regex = '<OPTION VALUE=\'(\\d+)\'><FONT SIZE=\+1>(.*?)</FONT>'
    mats = re.finditer(regex, page)
    # print mats
    clsIds = []
    for mat in mats:
        clsIds.append( (mat.group(1), mat.group(2)) )

    return clsIds

def getKeywords(kwId, outdir):
    tmpS = u'查詢'
    tmpS = tmpS.encode('big5')
    url = 'http://cls.hs.yzu.edu.tw/tang/MakePoem/qry_name.asp'
    values = {'TypeList':kwId, 'SELTYPE':tmpS}
    POST(url, values, os.path.join(outdir, kwId + '.html'))

def getNameList(aspPage):
    print(aspPage)
    page = open(aspPage).read().decode('big5', 'ignore')
    regex = '<OPTION VALUE=\'(\\d+)\'><FONT SIZE=\+1 COLOR=#FF00FF>(.*?)</FONT>'
    mats = re.finditer(regex, page)
    print(mats)
    namesIds = []
    for mat in mats:
        namesIds.append( (mat.group(1), mat.group(2)) )

    return namesIds

def getBody(nameId, outdir):
    tmpS = u'查詢'
    tmpS = tmpS.encode('big5')
    url = 'http://cls.hs.yzu.edu.tw/tang/MakePoem/qry_body.asp'
    values = {'NameList':nameId, 'SELTYPE':tmpS}
    POST(url, values, os.path.join(outdir, nameId + '.html'))

def extractPhrase(infile):
    page = open(infile).read().decode('big5', 'ignore')
    regex = '<TD class=font_purple>(.*?)</TD>'
    mats = re.finditer(regex, page)
    phrases = []
    for mat in mats:
        phrase = mat.group(1).strip()
        phrases.append(phrase)

    return phrases

def uwrite(fout, s):
    fout.write(s.encode('utf-8'))
    fout.flush()

def toStdFormat(aspPageF, classIdD, nameListD, outfile):
    fout = open(outfile, 'w')
    clsIds = getKeywordsIDs(aspPageF)
    for (i, (cId, name)) in enumerate(clsIds):
        print(i, cId, name)
        name = name.replace(u' ', '')
        uwrite(fout, '<begin>\t%d\t%s\n' % (i+1,name))
        keyWdsF = os.path.join(classIdD, cId + '.html')
        namesIds = getNameList(keyWdsF)
        for (j, (nid, nname)) in enumerate(namesIds):
            nname = nname.replace(u' ', '')
            uwrite(fout, '%d\t%s\t' % (j+1, nname))
            phraseF = os.path.join(nameListD, nid + '.html')
            phrases = extractPhrase(phraseF)
            uwrite(fout, ' '.join([phrase.replace(u' ', '') for phrase in phrases]))
            uwrite(fout, '\n')


        uwrite(fout, '<end>\t%d\n' % (i+1))

    fout.close()

    fan2jian4file(outfile, outfile + '.jian')


def step1():
    # step one: run the following code
    clsIds = getKeywordsIDs('qry_type.asp')
    print(clsIds)
    print(len(clsIds))
    for clsId in clsIds:
        getKeywords(clsId[0], 'classId')
    # make sure there is no exception before this. 
    # If there is, run the code above again.

def step2(beginId):
    # step two: run the following code
    # note that these code below always throw exceptions
    # due to the instability of your WLAN.
    # record the id of the failed page and restart step2 from that id
    begin = beginId == '0'
    print(beginId, begin)
    for f in os.listdir('classId'):
        if f.startswith('.'): continue
        fname = os.path.join('classId', f)
        namesIds = getNameList(fname)
        for nameId in namesIds:
            if nameId[0] == beginId: begin = True
            if begin:
                getBody(nameId[0], 'nameList')
                time.sleep(5)

def step3():
    # convert the crawled pages to standard format
    toStdFormat('qry_type.asp', 'classId', 'nameList', 'Shixuehanying.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl Shixuehanying. Note there are three steps, \nand you might need to run step2 multiple times!\nthe output file is Shixuehanying.txt and Shixuehanying.jian')
    parser.add_argument('step', help = 'the valid value is 1, 2 and 3')
    parser.add_argument('-b', '--begin', help = 'begin id used in step 2. If step2 failed with exception, try to start with the page id at which it failed. The id is the number after NameList. e.g. NameList=00000028&SELTYPE=... the id is 00000028', default = '0')
    args = parser.parse_args()
    if args.step == '1':
        step1()
    elif args.step == '2':
        step2(args.begin)
    elif args.step == '3':
        step3()
