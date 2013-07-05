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
import base64
import os
import sys
import cgi
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from compiler.ast import Print
wnl = nltk.WordNetLemmatizer()
enchant.set_param('enchant.myspell.dictionary.path',\
                   '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/enchant/share/enchant/myspell/')
PORT_NUMBER = 8080

QUERY_LEMMA_MATCH = 1
QUERY_SYNSET_MATCH = 2
QUERY_LEMMA_SYNSET_MATCH = 3

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

def ParseClickLogAndCreateNGrammsIndexes(click_log_file):
    fin = codecs.open(click_log_file, "r", "utf-8")

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
    
    return unigramm_index, bigramm_index, trigramm_index

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
            expanded_lemmas.add(l)
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

def CreateDicts(ngramms_in_index):
    lemma_ngramm_to_pics = {}
    lemma_pics_to_ngramm = {}
    synset_ngramm_to_pics = {}
    synset_pics_to_ngramm = {}
    lemma_synset_ngramm_to_pics = {}
    lemma_synset_pics_to_ngramm = {}

    
    for (ngramm, match_type, pics) in ngramms_in_index:
        if match_type == QUERY_LEMMA_MATCH:
            for pic in pics:
                PutToDicts(ngramm, pic, lemma_ngramm_to_pics, lemma_pics_to_ngramm)
        elif match_type == QUERY_SYNSET_MATCH:
            for pic in pics:
                PutToDicts(ngramm, pic, synset_ngramm_to_pics, synset_pics_to_ngramm)
        elif match_type == QUERY_LEMMA_SYNSET_MATCH:
            for pic in pics:
                PutToDicts(ngramm, pic, lemma_synset_ngramm_to_pics, lemma_synset_pics_to_ngramm)

                
    return lemma_ngramm_to_pics, lemma_pics_to_ngramm,\
         synset_ngramm_to_pics, synset_pics_to_ngramm,\
         lemma_synset_ngramm_to_pics, lemma_synset_pics_to_ngramm

def DrawPics(out, caption, pics_to_ngramms):
    out.write("<p>")
    out.write("<h1>%s</h1>" % caption)
    out.write("<table><tr>")
    for pic in pics_to_ngramms.keys():
        out.write("<td><img src=\"imageget/%s\"></td>" % escape_image_id_to_be_valid_filename(pic))
        out.write("<td>%d %s</td>" % (len(pics_to_ngramms[pic]), str(pics_to_ngramms[pic])))
    out.write("</tr></table>")

def CalcImageRelevance(out, query, image):
            
    query_lemmas = GetLemmas(query)
    out.write("<table><tr><td>%s</td><td>%s</td></tr></table><br/>" % ("query lemmas", str(query_lemmas)))
    
    expanded_lemmas = {}
    for lemma in query_lemmas:
        expanded_lemmas[lemma] = ExpandLemma(lemma)
    
    unigramms_to_analyze = ExpandUnigramms(query_lemmas, expanded_lemmas)
    bigramms_to_analyze = ExpandBigramms(query_lemmas, expanded_lemmas)    
    trigramms_to_analyze = ExpandTrigramms(query_lemmas, expanded_lemmas)

    unigramms_in_index = [(unigramm, match_type, unigramm_index[unigramm]) for (unigramm, match_type) in unigramms_to_analyze if unigramm in unigramm_index]
    bigramms_in_index = [(bigramm, match_type, bigramm_index[bigramm]) for (bigramm, match_type) in bigramms_to_analyze if bigramm in bigramm_index]
    trigramms_in_index = [(trigramm, match_type, trigramm_index[trigramm]) for (trigramm, match_type) in trigramms_to_analyze if trigramm in trigramm_index]
    
    lemma_unigramm_to_pics, lemma_pics_to_unigramm,\
    synset_unigramm_to_pics, synset_pics_to_unigramm,\
    lemma_synset_unigramm_to_pics, lemma_synset_pics_to_unigramm =\
        CreateDicts(unigramms_in_index)
        
    lemma_bigramm_to_pics, lemma_pics_to_bigramm,\
    synset_bigramm_to_pics, synset_pics_to_bigramm,\
    lemma_synset_bigramm_to_pics, lemma_synset_pics_to_bigramm =\
        CreateDicts(bigramms_in_index)

    lemma_trigramm_to_pics, lemma_pics_to_trigramm,\
    synset_trigramm_to_pics, synset_pics_to_trigramm,\
    lemma_synset_trigramm_to_pics, lemma_synset_pics_to_trigramm =\
        CreateDicts(trigramms_in_index)


    DrawPics(out, "Matching by unigramm lemma.", lemma_pics_to_unigramm)
    DrawPics(out, "Matching by unigramm synsetlemma.", synset_pics_to_unigramm)

    DrawPics(out, "Matching by bigramm lemma.", lemma_pics_to_bigramm)
    DrawPics(out, "Matching by bigramm lemma_synsetlemma.", lemma_synset_pics_to_bigramm)
    DrawPics(out, "Matching by bigramm synsetlemma.", synset_pics_to_bigramm)

    DrawPics(out, "Matching by trigramm lemma.", lemma_pics_to_trigramm)
    DrawPics(out, "Matching by trigramm lemma_synsetlemma.", lemma_synset_pics_to_trigramm)
    DrawPics(out, "Matching by trigramm synsetlemma.", synset_pics_to_trigramm)
    
    out.write("</p>")

    
    return np.random.randn()

if __name__ == '__main__':
    global unigramm_index
    global bigramm_index
    global trigramm_index
    global img_index
    unigramm_index, bigramm_index, trigramm_index = ParseClickLogAndCreateNGrammsIndexes(click_log_file)
    
    try:
        server = HTTPServer(('', PORT_NUMBER), MyHandler)
        print 'Started httpserver on port ' , PORT_NUMBER
        server.serve_forever()
    except KeyboardInterrupt:
        print '^C received, shutting down the web server'
        server.socket.close()