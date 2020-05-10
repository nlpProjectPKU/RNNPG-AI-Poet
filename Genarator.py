# -*- coding: utf-8 -*-

import FirstSenGenerator
import GeneratorFor5
import GeneratorFor7

FirstSenGenerator.find_best_sentences()
chars = 0
with open("./shengcheng/top.txt", 'r', encoding='utf-8') as f:
    a = f.readlines()
    chars = len(a[0]) - 1

if chars == 5:
    GeneratorFor5.print_topn()
else:
    GeneratorFor7.print_topn()