# -*- coding: utf-8 -*-

import FirstSenGenerator
import GeneratorFor5
import GeneratorFor7

chars = FirstSenGenerator.find_best_sentences()

if chars == 5:
    GeneratorFor5.print_topn()
else:
    GeneratorFor7.print_topn()
