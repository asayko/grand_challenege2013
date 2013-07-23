#!/usr/bin/env python

import codecs
import sys
import random
import numpy as np
import math
import httplib, urllib

labeled_queries_file = "/Users/asayko/data/grand_challenge/Dev/DevSetLabel.tsv"
images_file = "/Users/asayko/data/grand_challenge/Dev/DevSetImage.tsv"
server_path = "localhost:8080"
url_selector = "/imagerank"
n_random_queries = 10
n_random_pics = 10000

def LoadQueries(labeled_queries_file):
    labeled_queries = {}
    for line in codecs.open(labeled_queries_file, "r", "utf-8").xreadlines():
        line_parts = line.split("\t")
        query = line_parts[0].strip()
        img_id = line_parts[1].strip()
        label = line_parts[2].strip()
        labeled_queries.setdefault(query, []).append((img_id, label))
    return labeled_queries

def GetRelevanceForQuery(query, img_id, img_base64, server_path, url_selector, rel_label):
    params = urllib.urlencode({'runID': 0, 'query': query, 'image': img_base64, 'img_id': img_id, 'rel_label': rel_label})
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    conn = httplib.HTTPConnection(server_path)
    print >> sys.stderr, "Requesting %s%s for query %s with img %s:%s..." % (server_path, url_selector, query, img_id, img_base64[:10])
    conn.request("POST", url_selector, params, headers)
    response = conn.getresponse()
    print >> sys.stderr, response.status, response.reason
    data = response.read()
    print >> sys.stderr, data
    rel = float(data)
    return rel

def LoadImages(images_file):
    images = {}
    for line in codecs.open(images_file, "r", "utf-8").xreadlines():
        line_parts = line.split("\t")
        img_id = line_parts[0].strip()
        img_base64 = line_parts[1].strip()
        images[img_id] = img_base64
    return images

def LabelToFloat(label):
    if label == "Bad":
        return 0.0
    elif label == "Good":
        return 2.0
    elif label == "Excellent":
        return 3.0
    else:
        return 0.0 # or raise?

def CalcMetric(metric_source_data):
    sorted_metric_source_data = sorted([(rel, random.random(), label) for (rel, label) in metric_source_data])
    pos = 1
    metric = 0.0
    for (rel, r, label) in sorted_metric_source_data:
        metric = metric + (pow(2.0, LabelToFloat(label)) - 1) / (math.log(pos + 1, 2))
        pos = pos + 1
        if pos == 26: break
    metric = 0.01757 * metric
    return metric

if __name__ == '__main__':
    print >> sys.stderr, "Loading labels."
    labeled_queries = LoadQueries(labeled_queries_file)
    print >> sys.stderr, "Loading images."
    images = LoadImages(images_file)
    print >> sys.stderr, "Shooting."
    random.seed(2)
    
    metrics = []
    
    for query in random.sample(labeled_queries.keys(), min(n_random_queries, len(labeled_queries.keys()))):
        print >> sys.stderr, "Shooting for query %s." % query
        labels = labeled_queries[query]
        metric_source_data = []
        for (img_id, label) in random.sample(labels, min(n_random_pics, len(labels))):
            img_base64 = images[img_id]
            rel = GetRelevanceForQuery(query, img_id, img_base64, server_path, url_selector, label)
            metric_source_data.append((rel, label))
        print metric_source_data
        metric = CalcMetric(metric_source_data)
        metrics.append(metric)
        mean_metric = np.mean(metrics)
        std_err_metric = np.std(metrics)
        print >> sys.stderr, "Current mean metric is %lf" % mean_metric
        print >> sys.stderr, "Current std err of metric is %lf" % std_err_metric
        
    mean_metric = np.mean(metrics)
    std_err_metric = np.std(metrics)
    
    print >> sys.stderr, "Final mean metric is %lf" % mean_metric
    print >> sys.stderr, "Final std err of metric is %lf" % std_err_metric
    
