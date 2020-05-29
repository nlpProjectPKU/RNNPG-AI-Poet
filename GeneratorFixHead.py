# -*- coding: utf-8 -*-

import GeneratorFor5_FixedHead
import GeneratorFor7_FixedHead

chars = 0
while True:  # choose 5 or 7
    print("Please choose poem structure:\npress 5: 5-char quatrain\npress 7: 7-char quatrain")
    chars = input()
    chars = int(chars)
    if chars == 5 or chars == 7:
        break
    else:
        print("Invalid input. Please try again.")

if chars == 5:
    GeneratorFor5_FixedHead.print_topn()
else:
    GeneratorFor7_FixedHead.print_topn()