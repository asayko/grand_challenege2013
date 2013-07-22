#!/usr/bin/env python

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import codecs
import itertools
import collections
import sys
import cPickle as pickle

wnl = nltk.WordNetLemmatizer()

click_log_file = "/Users/asayko/data/grand_challenge/Train/TrainClickLog100K.tsv"
visual_words_file = "./vis_words_10000.tsv"

query_index_save_to_file = "./query_index.pkl"
unigramm_index_save_to_file = "./unigramm_index.pkl"
bigramm_index_save_to_file = "./bigramm_index.pkl"
trigramm_index_save_to_file = "./trigramm_index.pkl"
visual_words_save_to_file = "./vis_words_index.pkl"

min_clicks_needed = 1

dont_load_unigramms_for_img_ids_file = "./dont_load_unigramms_for_pics.txt"
dont_load_bigramms_for_img_ids_file = "./dont_load_bigramms_for_pics.txt"
dont_load_trigramms_for_img_ids_file = "./dont_load_trigramms_for_pics.txt"

stop_words = set(stopwords.words('english'))

images_stop_words = set(['photo', 'pic', 'image', 'picture', 'free', 'video', 'photes', 'pichures', '1024',\
                          'iamges', 'picturers', 'picuters', 'pictires', 'picure', 'imagens', 'picter', '1920x1080', 'imags',\
                           'pcitures', 'imeges', 'pitcures', 'pictrues', 'imges', 'pictuer', 'pictur', 'imiges',\
                           'picutres', 'imagenes', 'picures', 'pictues','pictuers', 'photography', 'pitures',\
                           'pitcure', 'picftures', 'picitures', 'picters'])
unigramm_stop_words = set(['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',\
                            '2010', '2011', '2012', '2013', '2014', 'dr', 'pdf', 'jpeg', 'jpg', 'com', 'www'])

def GetLemmas(query):
    query.replace(".", " ")
    query_tokens = nltk.word_tokenize(query)
    query_lemmas = []
    for token in query_tokens:
        lemma = wnl.lemmatize(token)
        if lemma in stop_words: continue
        if lemma in images_stop_words: continue
        if not lemma.isalnum(): continue
        if not is_ascii(lemma): continue
        query_lemmas.append(lemma.lower())
    return query_lemmas

def QueryLemmasToNormalizedQuery(query_lemmas):
    return " ".join(sorted([l for l in query_lemmas if l not in stop_words and l not in images_stop_words]))

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def PutToIndex(index, key, img_id, num_clicks):
    counters = index.setdefault(key, {}).setdefault(img_id, [0, 0]) # [num_total_clicks, num_total_queries]
    counters[0] += num_clicks
    counters[1] += 1

def ParseClickLogAndCreateNGrammsIndexes(click_log_file):
    fin = codecs.open(click_log_file, "r", "utf-8")

    query_index = {}
    unigramm_index = {}
    bigramm_index = {}
    trigramm_index = {}
    
    print >> sys.stderr,  "Reading click log from %s" % click_log_file
      
    num_lines_processed = 0
    for line in fin.xreadlines():
        if num_lines_processed % 5000 == 0: print >> sys.stderr,  "%d lines processed" % num_lines_processed
        line_parts = line.split("\t")
        img_id = line_parts[0].strip()
        query = line_parts[1].strip()
        clicks_num = int(line_parts[2].strip())
        
        if clicks_num < min_clicks_needed:
            num_lines_processed = num_lines_processed + 1
            continue
        
        query_lemmas = GetLemmas(query)
        normalized_query = QueryLemmasToNormalizedQuery(query_lemmas)

        PutToIndex(query_index, normalized_query, img_id, clicks_num)

        if img_id not in dont_load_unigramms_for_img_ids:
            for lemma in query_lemmas:
                if lemma in unigramm_stop_words: continue
                unigramm = "%s" % lemma
                PutToIndex(unigramm_index, unigramm, img_id, clicks_num)

        if img_id not in dont_load_bigramms_for_img_ids:
            for bigramm in itertools.combinations(query_lemmas, 2):
                b = sorted(bigramm)
                bigramm = "%s %s" % (b[0], b[1])
                PutToIndex(bigramm_index, bigramm, img_id, clicks_num)
        
        if img_id not in dont_load_trigramms_for_img_ids:
            for trigramm in itertools.combinations(query_lemmas, 3):
                t = sorted(trigramm)
                trigramm = "%s %s %s" % (t[0], t[1], t[2])
                PutToIndex(trigramm_index, trigramm, img_id, clicks_num)
        num_lines_processed = num_lines_processed + 1
        
    return query_index, unigramm_index, bigramm_index, trigramm_index

def ParseVisualWordsBase(visual_words_file):
    fin = codecs.open(visual_words_file, "r", "utf-8")
    num_lines_processed = 0
    
    print >> sys.stderr,  "Reading visual words from %s" % visual_words_file
    
    vis_words_index = {}
    
    for line in fin.xreadlines():
        if num_lines_processed % 5000 == 0: print >> sys.stderr,  "%d lines processed" % num_lines_processed
        line_parts = line.split("\t")
        img_id = line_parts[0].strip()
        vis_words_str = line_parts[1].strip()
        vis_words = collections.Counter([int(w) for w in vis_words_str.split()])
        vis_words_index[img_id] = vis_words
        num_lines_processed = num_lines_processed + 1

    return vis_words_index


if __name__ == "__main__":
    
    dont_load_unigramms_for_img_ids = set([t.strip() for t in codecs.open(dont_load_unigramms_for_img_ids_file, "r", "utf-8")])
    dont_load_bigramms_for_img_ids = set([t.strip() for t in codecs.open(dont_load_bigramms_for_img_ids_file, "r", "utf-8")])
    dont_load_trigramms_for_img_ids = set([t.strip() for t in codecs.open(dont_load_trigramms_for_img_ids_file, "r", "utf-8")])
    
    #visual_words_index = ParseVisualWordsBase(visual_words_file)
    
    #print >> sys.stderr, "Saving %s" % visual_words_save_to_file
    #pickle.dump(visual_words_index, open(visual_words_save_to_file, "wb"))
    
    query_index, unigramm_index, bigramm_index, trigramm_index = ParseClickLogAndCreateNGrammsIndexes(click_log_file)

    print >> sys.stderr, "Saving %s" % query_index_save_to_file
    pickle.dump(query_index, open(query_index_save_to_file, "wb"))
    
    print >> sys.stderr, "Saving %s" % unigramm_index_save_to_file
    pickle.dump(unigramm_index, open(unigramm_index_save_to_file, "wb"))
    
    print >> sys.stderr, "Saving %s" % bigramm_index_save_to_file
    pickle.dump(unigramm_index, open(bigramm_index_save_to_file, "wb"))

    print >> sys.stderr, "Saving %s" % trigramm_index_save_to_file
    pickle.dump(unigramm_index, open(trigramm_index_save_to_file, "wb"))


