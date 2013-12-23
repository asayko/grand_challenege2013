#!/usr/bin/env python

from decaf.scripts.imagenet import DecafNet
import multiprocessing
import itertools
import codecs
import numpy as np
import sys
import urllib2
import Image
import io

def load_and_resize(stream):
    readed = Image.open(stream)
    (w, h) = readed.size
    M = min(w, h)
    frame_size = 227
    if M > frame_size:
        k = 227.0 / M;
        new_shape = (int(k * w), int(k * h))
        resized = readed.resize(new_shape)
    else:
        resized = readed
    return resized

def download_url(args):
    (url, net, queue) = args
    
    try:
        data = urllib2.urlopen(url, timeout = 10).read()
    except Exception as e:
        print >> sys.stderr, "Problem with loading image from %s" % url
        print >> sys.stderr, e
        queue.put((url, None, None))
        return

    try:
        stream = io.BytesIO(data)
        img = np.asarray(load_and_resize(stream))
        scores = net.classify(img, center_only=True)
        feature = net.feature('fc6_cudanet_out')
#        np.save(out_file, feature)
#        resized.save(os.path.join(out_folder, "%d.jpg" % i))

        #pil_image.thumbnail(thumb_size, Image.ANTIALIAS)
        #b64_image = base64.b64encode(data)
    
        #np_image = np.array(pil_image)
    
        #detector = cv2.SIFT()
        #key_points = detector.detect(np_image)
    
        #num_of_key_points = len(key_points)
    except Exception as e:
        print >> sys.stderr, "Problem with processing image from %s" % url
        print >> sys.stderr, e
        queue.put((url, None))
        return
    
    queue.put((url, feature))

def write_from_queue(queue, output_file):
    i = 0
    with codecs.open(output_file, 'w', 'utf-8') as fout:
        while True:
            try:
                res = queue.get()
                if res is None:
                    return
            
                i += 1
            
                (url, feature) = res
            
                print >> fout, "%s\t%s" % (url.strip().decode('utf-8'), "\t".join(map(str, feature[0])))
            
                if i % 10 == 0:
                    print >> sys.stderr, "%d of urls are processed" % i
            except Exception as e:
                print >> sys.stderr, "Problem with writing results for %s" % url
                print >> sys.stderr, e

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: %s NUM_OF_PROCESSES INPUT_URLS_LIST_FILE OUTPUT_FILE" % sys.argv[0]
        exit(1)
    
    N = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    net = DecafNet('../imagenet_pretrained/imagenet.decafnet.epoch90', '../imagenet_pretrained/imagenet.decafnet.meta')
    pool = multiprocessing.Pool(processes = N)
    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    
    p = multiprocessing.Process(target = write_from_queue, args = (results_queue, output_file,))
    p.start()
    
    pool.map(download_url, itertools.izip(codecs.open(input_file, 'r', 'utf-8').xreadlines(), itertools.repeat(net), itertools.repeat(results_queue)))
    pool.close()
    pool.join()
    
    print >> sys.stderr, "Sending None to the writer process."
    results_queue.put(None)
    p.join()
    print >> sys.stderr, "Done."

