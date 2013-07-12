#!/usr/bin/env python

import numpy as np
import scipy as sp
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import enchant
import codecs
import itertools
import random
import base64
import os
import sys
import cgi
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from compiler.ast import Print
wnl = nltk.WordNetLemmatizer()
enchant.set_param('enchant.myspell.dictionary.path',\
                   '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/enchant/share/enchant/myspell/')
dict_for_spellchecking = enchant.Dict("en_US")

PORT_NUMBER = 8080

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

MIN_POSSIBLE_VISUAL_MODEL_SIZE = 3
ENOUGH_VISUAL_MODEL_SIZE = 100

click_log_file = "/Users/asayko/data/grand_challenge/Train/TrainClickLog100K.tsv"
click_images_dir = "/Users/asayko/data/grand_challenge/Train/images_jpeg_renamed/"


dont_load_unigramms_for_img_ids_file = "/Users/asayko/workspace/grand2013challenge/dont_load_unigramms_for_pics.txt"
dont_load_unigramms_for_img_ids = set([t.strip() for t in codecs.open(dont_load_unigramms_for_img_ids_file, "r", "utf-8")])

dont_load_bigramms_for_img_ids_file = "/Users/asayko/workspace/grand2013challenge/dont_load_bigramms_for_pics.txt"
dont_load_bigramms_for_img_ids = set([t.strip() for t in codecs.open(dont_load_bigramms_for_img_ids_file, "r", "utf-8")])

dont_load_trigramms_for_img_ids_file = "/Users/asayko/workspace/grand2013challenge/dont_load_trigramms_for_pics.txt"
dont_load_trigramms_for_img_ids = set([t.strip() for t in codecs.open(dont_load_trigramms_for_img_ids_file, "r", "utf-8")])

stop_words = set(stopwords.words('english'))

images_stop_words = set(['photo', 'pic', 'image', 'picture', 'free', 'video', 'photes', 'pichures', '1024',\
                          'iamges', 'picturers', 'picuters', 'pictires', 'picure', 'imagens', 'picter', '1920x1080', 'imags',\
                           'pcitures', 'imeges', 'pitcures', 'pictrues', 'imges', 'pictuer', 'pictur', 'imiges',\
                           'picutres', 'imagenes', 'picures', 'pictues','pictuers', 'photography', 'pitures',\
                           'pitcure', 'picftures', 'picitures', 'picters'])
unigramm_stop_words = set(['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',\
                            '2010', '2011', '2012', '2013', '2014', 'dr', 'pdf', 'jpeg', 'jpg', 'com', 'www'])

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
        self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write("<html><body>")
        if self.path == "/imagerank":
            #if not form.has_key("runID") or not form.has_key("query") or not form.has_key("image"):
            if not form.has_key("query"):
                self.wfile.write("Missing required parameter.")
            else:
                self.PrintOutInputForm()
                query = form.getvalue("query", "default")
                image = form.getvalue("image", "default")
                relev = CalcImageRelevance(self.wfile, query, image)
                self.wfile.write(relev)
        self.PrintOutInputForm()
        self.wfile.write("</body></html>")
        return

def escape_image_id_to_be_valid_filename(image_id):
    return base64.urlsafe_b64encode(image_id)

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def PutToIndex(index, key, img_id): 
    if index.has_key(key):
        index[key].add(img_id)
    else:
        index[key] = set()
        index[key].add(img_id)

def QueryLemmasToNormalizedQuery(query_lemmas):
    return " ".join(sorted([l for l in query_lemmas if l not in stop_words and l not in images_stop_words]))

def ParseClickLogAndCreateNGrammsIndexes(click_log_file):
    fin = codecs.open(click_log_file, "r", "utf-8")

    query_index = {}
    unigramm_index = {}
    bigramm_index = {}
    trigramm_index = {}
      
    num_lines_processed = 0
    for line in fin.xreadlines():
        if num_lines_processed % 5000 == 0: print >> sys.stderr,  "%d lines processed" % num_lines_processed
        line_parts = line.split("\t")
        img_id = line_parts[0].strip()
        query = line_parts[1].strip()
        clicks_num = int(line_parts[2].strip())
    
        query_lemmas = GetLemmas(query)
        normalized_query = QueryLemmasToNormalizedQuery(query_lemmas)

        PutToIndex(query_index, normalized_query, img_id)

        if img_id not in dont_load_unigramms_for_img_ids:
            for lemma in query_lemmas:
                if lemma in unigramm_stop_words: continue
                unigramm = "%s" % lemma
                PutToIndex(unigramm_index, unigramm, img_id)

        if img_id not in dont_load_bigramms_for_img_ids:
            for bigramm in itertools.combinations(query_lemmas, 2):
                b = sorted(bigramm)
                bigramm = "%s %s" % (b[0], b[1])
                PutToIndex(bigramm_index, bigramm, img_id)
        
        if img_id not in dont_load_trigramms_for_img_ids:
            for trigramm in itertools.combinations(query_lemmas, 3):
                t = sorted(trigramm)
                trigramm = "%s %s %s" % (t[0], t[1], t[2])
                PutToIndex(trigramm_index, trigramm, img_id)    
        num_lines_processed = num_lines_processed + 1
    
    return query_index, unigramm_index, bigramm_index, trigramm_index

def GetLemmas(query):
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

def PutToDicts(ngramm, pic, ngramm_to_pics, pics_to_ngramm):
    if pic in pics_to_ngramm:
        pics_to_ngramm[pic].add(ngramm)
    else:
        pics_to_ngramm[pic] = set()
        pics_to_ngramm[pic].add(ngramm)
        
    if ngramm in ngramm_to_pics:
        ngramm_to_pics[ngramm].add(pic)
    else:
        ngramm_to_pics[ngramm] = set()
        ngramm_to_pics[ngramm].add(pic)

def DrawPics(out, caption, pics_to_ngramms):
    out.write("<p>")
    out.write("<h1>%s</h1>" % caption)
    out.write("<h3>%d images totally</h3>" % len(pics_to_ngramms.keys()))
    out.write("<table>")
    keys_to_draw = random.sample(pics_to_ngramms.keys(), min(len(pics_to_ngramms.keys()), 50))     
    for pic in keys_to_draw:
        out.write("<tr>")
        out.write("<td><img src=\"imageget/%s\"></td>" % escape_image_id_to_be_valid_filename(pic))
        out.write("<td>%d %s</td>" % (len(pics_to_ngramms[pic]), str(pics_to_ngramms[pic])))
        out.write("</tr>")
    out.write("</table>")

def CreateVisualModelForQuery(out, query):
    # trying to collect some how relevant pics from click_log_db  
    query_visual_model = {}

    query_lemmas = GetLemmas(query)
    normalized_query = QueryLemmasToNormalizedQuery(query_lemmas)

    out.write("<table><tr><td>%s</td><td>%s</td></tr></table><br/>" % ("query:", query))
    out.write("<table><tr><td>%s</td><td>%s</td></tr></table><br/>" % ("query lemmas:", str(query_lemmas)))
    out.write("<table><tr><td>%s</td><td>%s</td></tr></table><br/>" % ("normalized query:", normalized_query))

    expanded_lemmas = {}
    for lemma in query_lemmas:
        expanded_lemmas[lemma] = ExpandLemma(lemma)
    
    unigramms_to_analyze = ExpandUnigramms(query_lemmas, expanded_lemmas)
    bigramms_to_analyze = ExpandBigramms(query_lemmas, expanded_lemmas)    
    trigramms_to_analyze = ExpandTrigramms(query_lemmas, expanded_lemmas)

    # adding norm query matches
    for pic in query_index.get(normalized_query, set()):
         query_visual_model.setdefault(pic, set()).add(QUERY_NORMALIZED_QUERY_MATCH)
    
    # ading trigramms lemma and synset_lemma matches
    for (trigramm, match_type) in trigramms_to_analyze:
        if trigramm in trigramm_index and match_type == QUERY_LEMMA_MATCH:
            for pic in trigramm_index[trigramm]:
                query_visual_model.setdefault(pic, set()).add(QUERY_TRIGRAMM_LEMMA_MATCH)
        elif trigramm in trigramm_index and match_type == QUERY_LEMMA_SYNSET_MATCH:
            for pic in trigramm_index[trigramm]:
                query_visual_model.setdefault(pic, set()).add(QUERY_TRIGRAMM_LEMMA_SYNSET_MATCH)
                
    # adding bigramms lemma matches
    for (bigramm, match_type) in bigramms_to_analyze:
        if bigramm in bigramm_index and match_type == QUERY_LEMMA_MATCH:
             for pic in bigramm_index[bigramm]:
                 query_visual_model.setdefault(pic, set()).add(QUERY_BIGRAMM_LEMMA_MATCH)
    
    # if we have enough then stop
    if len(query_visual_model) > ENOUGH_VISUAL_MODEL_SIZE:
        return query_visual_model
    
    
    # add most valuable unigramms
    for (unigramm, match_type) in unigramms_to_analyze:
        if unigramm in unigramm_index and match_type == QUERY_LEMMA_MATCH:
            pos = nltk.pos_tag([unigramm])[-1][-1]
            if pos in noun_like_unigramm_pos_tags:
                for pic in unigramm_index[unigramm]:
                    query_visual_model.setdefault(pic, set()).add(QUERY_UNIGRAMM_LEMMA_NN_MATCH)
    
    # adding bigramms lemma synset matches
    for (bigramm, match_type) in bigramms_to_analyze:
        if bigramm in bigramm_index and match_type == QUERY_LEMMA_SYNSET_MATCH:
             for pic in bigramm_index[bigramm]:
                 query_visual_model.setdefault(pic, set()).add(QUERY_BIGRAMM_LEMMA_SYNSET_MATCH)

    # if we have enough then stop
    if len(query_visual_model) > ENOUGH_VISUAL_MODEL_SIZE:
        return query_visual_model

    # adding trigramms synset synset synset
    for (trigramm, match_type) in trigramms_to_analyze:
        if trigramm in trigramm_index and match_type == QUERY_SYNSET_MATCH:
            for pic in trigramm_index[trigramm]:
                query_visual_model.setdefault(pic, set()).add(QUERY_TRIGRAMM_SYNSET_MATCH)

    # adding bigramms lemma matches
    for (bigramm, match_type) in bigramms_to_analyze:
        if bigramm in bigramm_index and match_type == QUERY_SYNSET_MATCH:
             for pic in bigramm_index[bigramm]:
                 query_visual_model.setdefault(pic, set()).add(QUERY_BIGRAMM_SYNSET_MATCH)

    # if we have enough then stop
    if len(query_visual_model) > ENOUGH_VISUAL_MODEL_SIZE:
        return query_visual_model

    # add most valuable synset unigramms
    for (unigramm, match_type) in unigramms_to_analyze:
        if unigramm in unigramm_index and match_type == QUERY_SYNSET_MATCH:
            pos = nltk.pos_tag([unigramm])[-1][-1]
            if pos in noun_like_unigramm_pos_tags:
                for pic in unigramm_index[unigramm]:
                    query_visual_model.setdefault(pic, set()).add(QUERY_UNIGRAMM_SYNSET_NN_MATCH)    

    # if we have enough then stop
    if len(query_visual_model) > ENOUGH_VISUAL_MODEL_SIZE:
        return query_visual_model

    # add all lemma unigramms
    for (unigramm, match_type) in unigramms_to_analyze:
        if unigramm in unigramm_index and match_type == QUERY_LEMMA_MATCH:
            for pic in unigramm_index[unigramm]:
                query_visual_model.setdefault(pic, set()).add(QUERY_UNIGRAMM_LEMMA_MATCH)

    # add all synset unigramms
    for (unigramm, match_type) in unigramms_to_analyze:
        if unigramm in unigramm_index and match_type == QUERY_SYNSET_MATCH:
            for pic in unigramm_index[unigramm]:
                query_visual_model.setdefault(pic, set()).add(QUERY_UNIGRAMM_SYNSET_MATCH)    
            
    return query_visual_model

def SpellCheckingEnrich(query):
    query_lemmas = GetLemmas(query)
    enriched_lemmas = []
    
    for l in query_lemmas:
        enriched_lemmas.append(l)
        if not dict_for_spellchecking.check(l):
            suggested_lemmas = dict_for_spellchecking.suggest(l)
            for sl in suggested_lemmas:
                enriched_lemmas.append(sl)
                
    return " ".join(enriched_lemmas)

def CalcImageRelevance(out, query, image):
                    
    query_visual_model = CreateVisualModelForQuery(out, query)
    
    if len(query_visual_model) < MIN_POSSIBLE_VISUAL_MODEL_SIZE:
        query = SpellCheckingEnrich(query)
        query_visual_model = CreateVisualModelForQuery(out, query)
    
    DrawPics(out, "Visual query model.", query_visual_model)
    
    return np.random.randn()

if __name__ == '__main__':
    global query_index
    global unigramm_index
    global bigramm_index
    global trigramm_index
    query_index, unigramm_index, bigramm_index, trigramm_index = ParseClickLogAndCreateNGrammsIndexes(click_log_file)
    
    try:
        server = HTTPServer(('', PORT_NUMBER), MyHandler)
        print 'Started httpserver on port ' , PORT_NUMBER
        server.serve_forever()
    except KeyboardInterrupt:
        print '^C received, shutting down the web server'
        server.socket.close()