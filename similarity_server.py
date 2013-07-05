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

PORT_NUMBER = 8080

wnl = nltk.WordNetLemmatizer()
enchant.set_param('enchant.myspell.dictionary.path',\
                   '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/enchant/share/enchant/myspell/')
click_log_file = "/Users/asayko/data/grand_challenge/Train/TrainClickLog.tsv"
stop_words = set(stopwords.words('english'))
images_stop_words = set(['photo', 'pic', 'image', 'picture', 'free', 'video', 'photes', 'pichures', '1024',\
                          'iamges', 'picturers', 'picuters', 'pictires', 'picure', 'imagens', 'picter', '1920x1080', 'imags',\
                           'pcitures', 'imeges', 'pitcures', 'pictrues', 'imges', 'pictuer', 'pictur', 'imiges',\
                           'picutres', 'imagenes', 'picures', 'pictues','pictuers', 'photography', 'pitures',\
                           'pitcure', 'picftures', 'picitures', 'picters'])
unigramm_stop_words = set(['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',\
                            '2010', '2011', '2012', '2013', '2014', 'dr', 'pdf', 'jpeg', 'jpg', 'com', 'www'])

class MyHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        
        if self.path.startswith("/imageget/"):
            path_parts = self.path.split("/")
            file_name = path_parts[-1]
            f = open("/Users/asayko/data/grand_challenge/Train/images_jpeg_renamed/" + file_name, 'rb')
            self.wfile.write(f.read())
            return
        
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        
        self.wfile.write("<html><body>")
        self.wfile.write("<form action=\"imagerank\" method=\"post\">")
        
        self.wfile.write("Run id: <input type=\"text\" name=\"runID\"/> <br/>")
        self.wfile.write("Query: <input type=\"text\" name=\"query\"/> <br/>")
        self.wfile.write("Image: <input type=\"text\" name=\"image\"/> <br/>")
        
        self.wfile.write("<input type=\"submit\" value=\"submit\"> <br/>")
        
        self.wfile.write("</form>")
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
        
        if self.path == "/imagerank":
            if not form.has_key("runID") or not form.has_key("query") or not form.has_key("image"):
                self.wfile.write("Missing required parameter.")
            else:
                query = form.getvalue("query", "default")
                image = form.getvalue("image", "default")
                relev = CalcImageRelevance(self.wfile, query, image)
                #self.wfile.write(relev)
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
    
    img_index = {}
    
    num_lines_processed = 0
    for line in fin.xreadlines():
        if num_lines_processed % 5000 == 0: print >> sys.stderr,  "%d lines processed" % num_lines_processed
        line_parts = line.split("\t")
        img_id = line_parts[0].strip()
        query = line_parts[1].strip()
        clicks_num = int(line_parts[2].strip())
    
        query_lemmas = GetLemmas(query)

        """
        for lemma in query_lemmas:
            if lemma in unigramm_stop_words: continue
            unigramm = "%s" % lemma
            #PutToIndex(unigramm_index, unigramm, img_id)
        """

        for bigramm in itertools.combinations(query_lemmas, 2):
            b = sorted(bigramm)
            bigramm = "%s %s" % (b[0], b[1])
            #PutToIndex(bigramm_index, bigramm, img_id)
            PutToIndex(img_index, img_id, bigramm)

        """
        for trigramm in itertools.combinations(query_lemmas, 3):
            t = sorted(trigramm)
            trigramm = "%s %s %s" % (t[0], t[1], t[2])
            PutToIndex(trigramm_index, trigramm, img_id)
        """         
        num_lines_processed = num_lines_processed + 1
    
    return unigramm_index, bigramm_index, trigramm_index, img_index

def GetLemmas(query):
    query_tokens = nltk.word_tokenize(query)
    query_lemmas = []
    for token in query_tokens:
        lemma = wnl.lemmatize(token)
        if lemma in stop_words: continue
        if lemma in images_stop_words: continue
        if not lemma.isalnum(): continue
        if not is_ascii(lemma): continue
        query_lemmas.append(lemma)
    return query_lemmas

def ExpandLemma(lemma):
    expanded_lemmas = set()
    for s in wn.synsets(lemma):
        for l in s.lemma_names:
            expanded_lemmas.add(l)
    return expanded_lemmas

def ExpandUnigramms(query_lemmas, expanded_lemmas):
    unigramms_to_analyze = []
    for lemma in query_lemmas:
        unigramms_to_analyze.append((lemma, 1.0))
        for expanded_lemma in expanded_lemmas[lemma]:
            unigramms_to_analyze.append((expanded_lemma, 0.1))
    return unigramms_to_analyze

def ExpandBigramms(query_lemmas, expanded_lemmas):
    bigramms_to_analyze = []
    for b in itertools.combinations(query_lemmas, 2):
        bigramm = sorted(b)
        bigramm_text = "%s %s" % (bigramm[0], bigramm[1])
        bigramms_to_analyze.append((bigramm_text, 2.0))
        
        lemmas_first_expanded = expanded_lemmas[bigramm[0]]
        lemmas_second_expanded = expanded_lemmas[bigramm[1]]
        
        for (le1, le2) in itertools.product(lemmas_first_expanded, lemmas_second_expanded):
             be = sorted([le1, le2])
             be_text = "%s %s" % (be[0], be[1])
             bigramms_to_analyze.append((be_text, 0.2))
    return bigramms_to_analyze

def ExpandTrigramms(query_lemmas, expanded_lemmas):
    trigramms_to_analyze = []
    for t in itertools.combinations(query_lemmas, 3):
        trigramm = sorted([t[0], t[1], t[2]])
        trigramm_text = "%s %s %s" % (trigramm[0], trigramm[1], trigramm[2])
        trigramms_to_analyze.append((trigramm_text, 3.0))

        lemmas_first_expanded = expanded_lemmas[trigramm[0]]
        lemmas_second_expanded = expanded_lemmas[trigramm[1]]
        lemmas_third_expanded = expanded_lemmas[trigramm[2]]

        for (le1, le2, le3) in itertools.product(lemmas_first_expanded, lemmas_second_expanded, lemmas_third_expanded):
            te = sorted([le1, le2, le3])
            te_text = "%s %s %s" % (te[0], te[1], te[2])
            trigramms_to_analyze.append((te_text, 0.3))
    return trigramms_to_analyze

def CalcImageRelevance(out, query, image):
    query_lemmas = GetLemmas(query)
    
    expanded_lemmas = {}
    for lemma in query_lemmas:
        expanded_lemmas[lemma] = ExpandLemma(lemma)
    
    unigramms_to_analyze = ExpandUnigramms(query_lemmas, expanded_lemmas)
    bigramms_to_analyze = ExpandBigramms(query_lemmas, expanded_lemmas)    
    trigramms_to_analyze = ExpandTrigramms(query_lemmas, expanded_lemmas)
    
    out.write("<html><body>\n")        
    out.write("\n")
    out.write(unigramms_to_analyze)
    out.write("\n")

    out.write("\n")
    out.write(bigramms_to_analyze)
    out.write("\n")

    out.write("\n")
    out.write(trigramms_to_analyze)
    out.write("\n")
    
    relevant_weighted_pics = {}
    relevant_weighted_pics_ngrams = {}
    
    for (unigramm, weight) in unigramms_to_analyze:
        if unigramm in unigramm_index:
            out.write("%s in index\n" % unigramm)
            unigramm_pics = unigramm_index[unigramm]
            out.write(unigramm_pics)
            out.write("\n")
            for pic in unigramm_pics:
                if pic in relevant_weighted_pics:
                    relevant_weighted_pics[pic] = relevant_weighted_pics[pic] + weight
                    relevant_weighted_pics_ngrams[pic].add(unigramm)
                else:
                    relevant_weighted_pics[pic] = weight
                    relevant_weighted_pics_ngrams[pic] = set()
                    relevant_weighted_pics_ngrams[pic].add(unigramm)

    for (bigramm, weight) in bigramms_to_analyze:
        if bigramm in bigramm_index:
            out.write("%s in index\n" % bigramm)
            bigramm_pics = bigramm_index[bigramm]
            out.write(bigramm_pics)
            out.write("\n")
            for pic in bigramm_pics:
                if pic in relevant_weighted_pics:
                    relevant_weighted_pics[pic] = relevant_weighted_pics[pic] + weight
                    relevant_weighted_pics_ngrams[pic].add(bigramm)
                else:
                    relevant_weighted_pics[pic] = weight
                    relevant_weighted_pics_ngrams[pic] = set()
                    relevant_weighted_pics_ngrams[pic].add(bigramm)
                    
    for (trigramm, weight) in trigramms_to_analyze:
        if trigramm in trigramm_index:
            out.write("%s in index\n" % trigramm)
            trigramm_pics = trigramm_index[trigramm]
            out.write(trigramm_pics)
            out.write("\n")
            for pic in trigramm_pics:
                if pic in relevant_weighted_pics:
                    relevant_weighted_pics[pic] = relevant_weighted_pics[pic] + weight
                    relevant_weighted_pics_ngrams[pic].add(trigramm)
                else:
                    relevant_weighted_pics[pic] = weight
                    relevant_weighted_pics_ngrams[pic] = set()
                    relevant_weighted_pics_ngrams[pic].add(trigramm)
    
    out.write("<br>")
    out.write("\nweighted pics\n")
    out.write("<br>")
    out.write("<br>")
    out.write("<br>")
    for pic in relevant_weighted_pics.keys():
        out.write("pic: %s pic_name: %s weight: %lf\n" % (pic, escape_image_id_to_be_valid_filename(pic), relevant_weighted_pics[pic]))
        out.write("<br>")
        out.write("words: %s" % str(relevant_weighted_pics_ngrams[pic]))
        out.write("<br>")
        out.write("<img src=\"%s\"/>" % ("imageget/" + escape_image_id_to_be_valid_filename(pic) + ".jpeg"))
        out.write("<br>")
        out.write("\n")
    out.write("\n")
    out.write("<br>")
    out.write("</body></html>\n")
    return np.random.randn()

if __name__ == '__main__':
    global unigramm_index
    global bigramm_index
    global trigramm_index
    global img_index
    unigramm_index, bigramm_index, trigramm_index, img_index = ParseClickLogAndCreateNGrammsIndexes(click_log_file)
    
    for key in img_index.keys():
        print "%s\t%s.jpeg\t%d\t%s" % (key, escape_image_id_to_be_valid_filename(key), len(img_index[key]), str(img_index[key]))

'''
    try:
        server = HTTPServer(('', PORT_NUMBER), MyHandler)
        print 'Started httpserver on port ' , PORT_NUMBER
        server.serve_forever()
    except KeyboardInterrupt:
        print '^C received, shutting down the web server'
        server.socket.close()
'''