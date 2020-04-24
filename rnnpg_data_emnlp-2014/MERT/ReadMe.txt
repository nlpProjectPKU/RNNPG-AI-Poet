All the files in VALID and TEST have the same naming convention.
The following example explains what a file name means.

q5.1000.12.src and q5.1000.12.ref-68 are used together.
q5.1000.12.src is the 1st sentences of the 1000 5-char quatrains;
while q5.1000.12.ref-68 is the reference 2nd sentences for the 1st sentences
of the 1000 5-char quatrains. 'ref-68' means there are at most 68 reference
2nd sentences for a 1st sentence. In most cases, the reference sentence number
is much less than 68, say M, but for programming convience, we force the reference to
repeat the first reference 68-M times to make all the 1st sentences have the
equal number of reference sentences.
