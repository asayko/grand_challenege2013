#!/usr/bin/env python

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import enchant
import codecs
import itertools
import collections
import random
import base64
import os
import subprocess
import sys
import cgi
import cPickle as pickle
import datetime
import heapq
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

import create_pickle_indexes

enchant.set_param('enchant.myspell.dictionary.path',\
                   '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/enchant/share/enchant/myspell/')
dict_for_spellchecking = enchant.Dict("en_US")

click_images_dir = "/Users/asayko/data/grand_challenge/Train/images_jpeg_renamed/"

online_vis_words_extractor = subprocess.Popen('./online_vis_words_extractor', stdin = subprocess.PIPE, stdout = subprocess.PIPE)

#external_relevance_calcer = subprocess.Popen(['./histextract', '-v', 'vocabs/vocab_l2_32768.dat', '-p', '-b', '-m', '10000', '-c', 'cache'],\
#                                              stdin = subprocess.PIPE, stdout = subprocess.PIPE)

logs_dir = "./logs/"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)

PORT_NUMBER = 8080

MIN_POSSIBLE_VISUAL_MODEL_SIZE = 3
ENOUGH_VISUAL_MODEL_SIZE = 100

QUERY_NORMALIZED_QUERY_MATCH = 'QUERY_NORMALIZED_QUERY_MATCH'
QUERY_LEMMA_MATCH = 'QUERY_LEMMA_MATCH'
QUERY_SYNSET_MATCH = 'QUERY_SYNSET_MATCH'
QUERY_LEMMA_SYNSET_MATCH = 'QUERY_LEMMA_SYNSET_MATCH'
QUERY_TRIGRAMM_LEMMA_MATCH = 'QUERY_TRIGRAMM_LEMMA_MATCH'
QUERY_TRIGRAMM_LEMMA_SYNSET_MATCH = 'QUERY_TRIGRAMM_LEMMA_SYNSET_MATCH'
QUERY_TRIGRAMM_SYNSET_MATCH = 'QUERY_TRIGRAMM_SYNSET_MATCH'
QUERY_BIGRAMM_LEMMA_MATCH = 'QUERY_BIGRAMM_LEMMA_MATCH'
QUERY_BIGRAMM_LEMMA_SYNSET_MATCH = 'QUERY_BIGRAMM_LEMMA_SYNSET_MATCH'
QUERY_BIGRAMM_SYNSET_MATCH = 'QUERY_BIGRAMM_SYNSET_MATCH'
QUERY_UNIGRAMM_LEMMA_MATCH = 'QUERY_UNIGRAMM_LEMMA_MATCH'
QUERY_UNIGRAMM_LEMMA_NN_MATCH = 'QUERY_UNIGRAMM_LEMMA_MATCH'
QUERY_UNIGRAMM_LEMMA_SYNSET_MATCH = 'QUERY_UNIGRAMM_LEMMA_SYNSET_MATCH'
QUERY_UNIGRAMM_SYNSET_MATCH = 'QUERY_UNIGRAMM_SYNSET_MATCH'
QUERY_UNIGRAMM_SYNSET_NN_MATCH = 'QUERY_UNIGRAMM_SYNSET_MATCH'

noun_like_unigramm_pos_tags = set(['NN', 'NNS', 'NNP', 'NNPS', ])

class MyHandler(BaseHTTPRequestHandler):
    
    def PrintOutInputForm(self):
        self.wfile.write("<br/>")
        self.wfile.write("<form action=\"imagerank\" method=\"post\">")
        self.wfile.write("Run id: <input type=\"text\" name=\"runID\"/> <br/>")
        self.wfile.write("Query: <input type=\"text\" name=\"query\"/> <br/>")
        self.wfile.write("Image: <input type=\"text\" name=\"image\"/> <br/>")
        self.wfile.write("<input type=\"submit\" value=\"submit\"> <br/>")
        self.wfile.write("</form>")
        self.wfile.write("<br/>")
        return
 
    def do_GET(self):
        
        if self.path.startswith("/imageget/"):
            self.send_response(200)
            self.send_header('Content-type','image/jpeg')
            self.end_headers()
            
            path_parts = self.path.split("/")
            img_id = path_parts[-1]
            f = open("%s%s.jpeg" % (click_images_dir, img_id) , 'rb')
            self.wfile.write(f.read())
            return
        
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        
        self.wfile.write("<html><body>")   
        self.PrintOutInputForm()
        self.wfile.write("</body></html>")
        return
    
    def do_POST(self):
        form = cgi.FieldStorage(
                fp=self.rfile, 
                headers=self.headers,
                environ={'REQUEST_METHOD':'POST',
                         'CONTENT_TYPE':self.headers['Content-Type'],
            })
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        if self.path == "/imagerank":
            if not form.has_key("query") or not form.has_key("image"):
                self.wfile.write("Missing required parameter.")
            else:
                query = form.getvalue("query", "default")
                image = form.getvalue("image", "default")
                request_img_id = form.getvalue("img_id", "default")
                rel_label = form.getvalue("rel_label", "default")
                relev = CalcImageRelevance(query, image, request_img_id, rel_label)
                self.wfile.write(relev)
        return

def escape_image_id_to_be_valid_filename(image_id):
    return base64.urlsafe_b64encode(image_id)

def ExpandLemma(lemma):
    expanded_lemmas = set()
    for s in wn.synsets(lemma):
        for l in s.lemma_names:
            ts = l.split('_')
            for t in ts:
                expanded_lemmas.add(t.lower())
    if lemma in expanded_lemmas: expanded_lemmas.remove(lemma)
    return expanded_lemmas

def ExpandUnigramms(query_lemmas, expanded_lemmas):
    unigramms_to_analyze = []
    for lemma in query_lemmas:
        unigramms_to_analyze.append((lemma, QUERY_LEMMA_MATCH))
        for expanded_lemma in expanded_lemmas[lemma]:
            unigramms_to_analyze.append((expanded_lemma, QUERY_SYNSET_MATCH))
    return unigramms_to_analyze

def ExpandBigramms(query_lemmas, expanded_lemmas):
    bigramms_to_analyze = []
    for b in itertools.combinations(query_lemmas, 2):
        bigramm = sorted(b)
        bigramm_text = "%s %s" % (bigramm[0], bigramm[1])
        bigramms_to_analyze.append((bigramm_text, QUERY_LEMMA_MATCH))
        
        lemmas_first_expanded = expanded_lemmas[bigramm[0]]
        lemmas_second_expanded = expanded_lemmas[bigramm[1]]
        
        for (bi1, le2) in itertools.product([bigramm[0]], lemmas_second_expanded):
             be = sorted([bi1, le2])
             be_text = "%s %s" % (be[0], be[1])
             bigramms_to_analyze.append((be_text, QUERY_LEMMA_SYNSET_MATCH))
             
        for (le1, bi2) in itertools.product(lemmas_first_expanded, [bigramm[1]]):
             be = sorted([le1, bi2])
             be_text = "%s %s" % (be[0], be[1])
             bigramms_to_analyze.append((be_text, QUERY_LEMMA_SYNSET_MATCH))

        for (le1, le2) in itertools.product(lemmas_first_expanded, lemmas_second_expanded):
             be = sorted([le1, le2])
             be_text = "%s %s" % (be[0], be[1])
             bigramms_to_analyze.append((be_text, QUERY_SYNSET_MATCH))
    return bigramms_to_analyze

def ExpandTrigramms(query_lemmas, expanded_lemmas):
    trigramms_to_analyze = []
    for t in itertools.combinations(query_lemmas, 3):
        trigramm = sorted([t[0], t[1], t[2]])
        trigramm_text = "%s %s %s" % (trigramm[0], trigramm[1], trigramm[2])
        trigramms_to_analyze.append((trigramm_text, QUERY_LEMMA_MATCH))

        lemmas_first_expanded = expanded_lemmas[trigramm[0]]
        lemmas_second_expanded = expanded_lemmas[trigramm[1]]
        lemmas_third_expanded = expanded_lemmas[trigramm[2]]

        for (le1, le2, le3) in itertools.product([trigramm[0]], lemmas_second_expanded, lemmas_third_expanded):
            te = sorted([le1, le2, le3])
            te_text = "%s %s %s" % (te[0], te[1], te[2])
            trigramms_to_analyze.append((te_text, QUERY_LEMMA_SYNSET_MATCH))

        for (le1, le2, le3) in itertools.product(lemmas_first_expanded, [trigramm[1]], lemmas_third_expanded):
            te = sorted([le1, le2, le3])
            te_text = "%s %s %s" % (te[0], te[1], te[2])
            trigramms_to_analyze.append((te_text, QUERY_LEMMA_SYNSET_MATCH))

        for (le1, le2, le3) in itertools.product(lemmas_first_expanded, lemmas_second_expanded, [trigramm[2]]):
            te = sorted([le1, le2, le3])
            te_text = "%s %s %s" % (te[0], te[1], te[2])
            trigramms_to_analyze.append((te_text, QUERY_LEMMA_SYNSET_MATCH))

        for (le1, le2, le3) in itertools.product([trigramm[0]], [trigramm[1]], lemmas_third_expanded):
            te = sorted([le1, le2, le3])
            te_text = "%s %s %s" % (te[0], te[1], te[2])
            trigramms_to_analyze.append((te_text, QUERY_LEMMA_SYNSET_MATCH))

        for (le1, le2, le3) in itertools.product([trigramm[0]], lemmas_second_expanded, [trigramm[2]]):
            te = sorted([le1, le2, le3])
            te_text = "%s %s %s" % (te[0], te[1], te[2])
            trigramms_to_analyze.append((te_text, QUERY_LEMMA_SYNSET_MATCH))

        for (le1, le2, le3) in itertools.product(lemmas_first_expanded, [trigramm[1]], [trigramm[2]]):
            te = sorted([le1, le2, le3])
            te_text = "%s %s %s" % (te[0], te[1], te[2])
            trigramms_to_analyze.append((te_text, QUERY_LEMMA_SYNSET_MATCH))

        for (le1, le2, le3) in itertools.product(lemmas_first_expanded, lemmas_second_expanded, lemmas_third_expanded):
            te = sorted([le1, le2, le3])
            te_text = "%s %s %s" % (te[0], te[1], te[2])
            trigramms_to_analyze.append((te_text, QUERY_SYNSET_MATCH))
            
    return trigramms_to_analyze

def ClickCounterTuppleToEmpiricalRelevance(ngramm, match_type, cliks_num, queries_num):
    click_rel = np.log(cliks_num) * queries_num
    num_of_lemmas = len([w for w in ngramm.split()])
    num_of_nouns = NGrammToNumOfNouns(ngramm) + 1e-3

    if match_type == QUERY_NORMALIZED_QUERY_MATCH:
        return click_rel * 1e5 
    
    elif match_type == QUERY_TRIGRAMM_LEMMA_MATCH:
        return click_rel * 1e4
    elif match_type == QUERY_TRIGRAMM_LEMMA_SYNSET_MATCH:
        return click_rel * num_of_nouns * 1e3
    elif match_type == QUERY_TRIGRAMM_SYNSET_MATCH:
        return click_rel * num_of_nouns * 1e2
    
    elif match_type == QUERY_BIGRAMM_LEMMA_MATCH:
        return click_rel * num_of_nouns * 1e3
    elif match_type == QUERY_BIGRAMM_LEMMA_SYNSET_MATCH:
        return click_rel * num_of_nouns * 1e1
    elif match_type == QUERY_BIGRAMM_SYNSET_MATCH:
        return click_rel * num_of_nouns * 1e0

    elif match_type == QUERY_UNIGRAMM_LEMMA_MATCH:
        return click_rel * num_of_nouns * 1e-1
    
    elif match_type == QUERY_UNIGRAMM_SYNSET_MATCH:
        return click_rel * num_of_nouns * 1e-2

    #rel = -0.0591716 * pow(queries_num, 2) + 1.65680473 * queries_num - 1.59763314;
    return rel

def AddPicCounters(visual_query_model, pic_counters, ngramm, match_type): 
    for pic in pic_counters:
        cliks_num = pic_counters[pic][0]
        queries_num = pic_counters[pic][1]
        visual_query_model.setdefault(pic, 0.0)
        visual_query_model[pic] = visual_query_model[pic] + ClickCounterTuppleToEmpiricalRelevance(ngramm, match_type, cliks_num, queries_num)

def NGrammToNumOfNouns(ngramm):
    if ngramm in ngramm_ngramm_to_num_of_nouns:
        return ngramm_ngramm_to_num_of_nouns[ngramm]
    num_of_nouns = 0 
    for w in ngramm.split():
        if w in lemmas_to_pos_tag:
            if lemmas_to_pos_tag[w] in noun_like_unigramm_pos_tags:
                num_of_nouns = num_of_nouns + 1
        else:
            lemmas_to_pos_tag[w] = nltk.pos_tag([w])[-1][-1]
            if lemmas_to_pos_tag[w] in noun_like_unigramm_pos_tags:
                num_of_nouns = num_of_nouns + 1
    ngramm_ngramm_to_num_of_nouns[ngramm] = num_of_nouns
    return ngramm_ngramm_to_num_of_nouns[ngramm]

def CreateVisualModelForQuery(query):
    # trying to collect some how relevant pics from click_log_db  
    visual_query_model = {}
    global lemmas_to_pos_tag
    lemmas_to_pos_tag = {}
    global ngramm_to_num_of_nouns
    ngramm_to_num_of_nouns = {}

    query_lemmas = create_pickle_indexes.GetLemmas(query)
    normalized_query = create_pickle_indexes.QueryLemmasToNormalizedQuery(query_lemmas)

    expanded_lemmas = {}
    for lemma in query_lemmas:
        expanded_lemmas[lemma] = ExpandLemma(lemma)
    
    unigramms_to_analyze = ExpandUnigramms(query_lemmas, expanded_lemmas)
    bigramms_to_analyze = ExpandBigramms(query_lemmas, expanded_lemmas)    
    trigramms_to_analyze = ExpandTrigramms(query_lemmas, expanded_lemmas)

    # adding norm query matches
    AddPicCounters(visual_query_model, query_index.get(normalized_query, {}), normalized_query, QUERY_NORMALIZED_QUERY_MATCH)
    
    # adding trigramms
    for (trigramm, match_type) in trigramms_to_analyze:
        if trigramm in trigramm_index and match_type == QUERY_LEMMA_MATCH:
            AddPicCounters(visual_query_model, trigramm_index[trigramm], trigramm, QUERY_TRIGRAMM_LEMMA_MATCH)
        elif trigramm in trigramm_index and match_type == QUERY_LEMMA_SYNSET_MATCH:
            AddPicCounters(visual_query_model, trigramm_index[trigramm], trigramm, QUERY_TRIGRAMM_LEMMA_SYNSET_MATCH)
        elif trigramm in trigramm_index and match_type == QUERY_SYNSET_MATCH:
            AddPicCounters(visual_query_model, trigramm_index[trigramm], trigramm, QUERY_TRIGRAMM_SYNSET_MATCH)

    # adding bigramms
    for (bigramm, match_type) in bigramms_to_analyze:
        if bigramm in bigramm_index and match_type == QUERY_LEMMA_MATCH:
            AddPicCounters(visual_query_model, bigramm_index[bigramm], bigramm, QUERY_BIGRAMM_LEMMA_MATCH)
        elif bigramm in bigramm_index and match_type == QUERY_LEMMA_SYNSET_MATCH:
            AddPicCounters(visual_query_model, bigramm_index[bigramm], bigramm, QUERY_BIGRAMM_LEMMA_SYNSET_MATCH)
        elif bigramm in bigramm_index and match_type == QUERY_SYNSET_MATCH:
            AddPicCounters(visual_query_model, bigramm_index[bigramm], bigramm, QUERY_BIGRAMM_SYNSET_MATCH)
    
    # add most valuable unigramms
    for (unigramm, match_type) in unigramms_to_analyze:
        if unigramm in unigramm_index and match_type == QUERY_LEMMA_MATCH:
            AddPicCounters(visual_query_model, unigramm_index[unigramm], unigramm, QUERY_UNIGRAMM_LEMMA_MATCH)
        elif unigramm in unigramm_index and match_type == QUERY_SYNSET_MATCH:
            AddPicCounters(visual_query_model, unigramm_index[unigramm], unigramm, QUERY_UNIGRAMM_SYNSET_MATCH)
            
    return visual_query_model

def SpellCheckingEnrich(query):
    query_lemmas = create_pickle_indexes.GetLemmas(query)
    enriched_lemmas = []
    
    for l in query_lemmas:
        enriched_lemmas.append(l)
        if not dict_for_spellchecking.check(l):
            suggested_lemmas = dict_for_spellchecking.suggest(l)
            for sl in suggested_lemmas:
                enriched_lemmas.append(sl)
                
    return " ".join(enriched_lemmas)

def ExtractVisualWords(image):
    online_vis_words_extractor.stdin.write("%s\n" % image)
    vis_words_line = online_vis_words_extractor.stdout.readline().strip()
    bag_of_vis_words = sorted([int(w) for w in vis_words_line.split()])
    return bag_of_vis_words    

def DumpVisualModel(query, image, visual_model, request_img_id, rel_label, pretty_logs = False):
    
    file_name = logs_dir + create_pickle_indexes.QueryLemmasToNormalizedQuery(create_pickle_indexes.GetLemmas(query))
    file_name += str(random.randint(0, 100000))
    #file_name = "data_for_external_relevance_calcer"
    fout = codecs.open(file_name, "w", "utf-8")
    
    if pretty_logs:
        print fout, "<html><body><p>"
    
    print >> fout, "%s img_id: %s, img_file:%s.jpeg Relevance: %s" % (query, request_img_id, escape_image_id_to_be_valid_filename(request_img_id), rel_label)

    if pretty_logs:
        print fout, "</p><p>"

    print >> fout, image

    if pretty_logs:
        print fout, "</p><p><table>"
        
    for pic, rel in visual_model:
        if pretty_logs:
            print >> fout, '<tr><td><img src="http://imgrover2.yandex.ru:8080/imageget/%s"></td><td> %s, %lf</td></tr>' % (escape_image_id_to_be_valid_filename(pic), pic, rel)
        else:
            print >> fout, "%s.jpeg, %s, %lf" % (escape_image_id_to_be_valid_filename(pic), pic, rel)

    if pretty_logs:
        print fout, "</table></p></body><html>\n"

    fout.close()
    return file_name

def CalcExternalRelevance(model_file):
    #external_relevance_calcer.stdin.write("%s\n" % model_file)
    #rel = float(external_relevance_calcer.stdout.readline().strip())
    return 0.777

def CalcImageRelevance(query, image, request_img_id, rel_label):
    
    #image_bag_of_visual_words = ExtractVisualWords(image)
    
    query_visual_model = CreateVisualModelForQuery(query)
    
    if len(query_visual_model) < MIN_POSSIBLE_VISUAL_MODEL_SIZE:
        query = SpellCheckingEnrich(query)
        query_visual_model = CreateVisualModelForQuery(out, query)
    
    query_visual_model_trunkated = heapq.nlargest(min(ENOUGH_VISUAL_MODEL_SIZE, len(query_visual_model)),\
                                                  query_visual_model.items(), key = lambda x: x[1])
    
    model_file = DumpVisualModel(query, image, query_visual_model_trunkated, request_img_id, rel_label)
    
    relevance = 0.0
    relevance = CalcExternalRelevance(model_file)
    
    """
    for clicked_pic in query_visual_model.keys():
        clicked_pic_vis_words = visual_words_index[clicked_pic]
        clicked_pic_vis_words_len = float(sum(clicked_pic_vis_words.values()))
        
        given_image_vis_words = collections.Counter(image_bag_of_visual_words)
        given_image_vis_words_len = float(len(image_bag_of_visual_words))
        
        intersection_len = float(sum((given_image_vis_words & clicked_pic_vis_words).values()))
        union_len = clicked_pic_vis_words_len + given_image_vis_words_len - intersection_len;
        
        if union_len != 0.0:
            relevance = relevance + pow(20, intersection_len / union_len) - 1.0
    """ 
    return relevance

#if __name__ == '__main__':
    #print >> sys.stderr, "Loading %s %s" % (create_pickle_indexes.visual_words_save_to_file, str(datetime.datetime.now()))
    #global visual_words_index
    #visual_words_index = pickle.load(open(create_pickle_indexes.visual_words_save_to_file, "rb"))

    print >> sys.stderr, "Loading %s %s" % (create_pickle_indexes.query_index_save_to_file, str(datetime.datetime.now()))
    global query_index
    query_index = pickle.load(open(create_pickle_indexes.query_index_save_to_file, "rb"))
    
    print >> sys.stderr, "Loading %s %s" % (create_pickle_indexes.unigramm_index_save_to_file, str(datetime.datetime.now()))
    global unigramm_index
    unigramm_index = pickle.load(open(create_pickle_indexes.unigramm_index_save_to_file, "rb"))
    
    print >> sys.stderr, "Loading %s %s" % (create_pickle_indexes.bigramm_index_save_to_file, str(datetime.datetime.now()))
    global bigramm_index
    bigramm_index = pickle.load(open(create_pickle_indexes.bigramm_index_save_to_file, "rb"))
    
    print >> sys.stderr, "Loading %s %s" % (create_pickle_indexes.trigramm_index_save_to_file, str(datetime.datetime.now()))
    global trigramm_index
    trigramm_index = pickle.load(open(create_pickle_indexes.trigramm_index_save_to_file))
        
    print >> sys.stderr, "Loading finished on %s" % str(datetime.datetime.now())

if __name__ == '__main__':
        
    try:
        server = HTTPServer(('', PORT_NUMBER), MyHandler)
        print 'Started httpserver on port ' , PORT_NUMBER
        server.serve_forever()
    except KeyboardInterrupt:
        print '^C received, shutting down the web server'
        server.socket.close()