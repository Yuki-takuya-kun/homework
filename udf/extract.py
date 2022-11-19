#!/usr/bin/env python
# -*- coding:utf-8 -*-
from deepdive import *
from stanfordcorenlp import StanfordCoreNLP


@tsv_extractor
@returns(lambda
         doc_id = 'text',
         sentence_id = 'text',
         sentence_text = 'text',
         tokens = 'text',
         pos_tags = 'text',
         ner_tags = 'text',
         :[])
def extract(doc_id = 'text', content = 'text'):
    count = 0
    nlp = StanfordCoreNLP('/deepdive/lib/', lang='zh')
    sentences = content.split('。')[:-1]
    sentences = [sentence + '。' for sentence in sentences]
    for sentence in sentences:
        tokens = nlp.word_tokenize(sentence)
        pos_tags = nlp.pos_tag(sentence)
        ner_tags = nlp.ner(sentence)
        count += 1
        yield [doc_id, doc_id+'_'+str(count), sentence, tokens, pos_tags, ner_tags]
        