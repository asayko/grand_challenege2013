import glob
import sys
import os.path
import numpy as np
import heapq

def L2(x,y):
    return ((x - y) ** 2).sum()

class SimSearch(object):
    index = []
    fnames = []
    def search(self, query, n):
        res = []
        for i,x in enumerate(self.index):
            res.append((L2(x, query), self.fnames[i]))
        n = min(n, len(res))
        print res
        return heapq.nsmallest(n, res)
        

    def __init__(self, folder):
        file_list = glob.glob(os.path.join(folder, "*.npy"))
        for fl in file_list:
            print fl
            self.index.append(np.array(np.load(fl)))
            self.fnames.append(os.path.basename(fl).split(".")[0] + ".jpg")
            

def draw_html(results, html):
    fhtml = open(html, 'w')
    fhtml.write("<html><style>td {vertical-align:top;}img {max-width:320px;max-height:320}</style>")
    for result in results:
        fhtml.write("<img src=%s>%f<br>\n" % (result[1], result[0]))
    fhtml.write("</html>")

if __name__ == '__main__':
    simSearch = SimSearch(sys.argv[1])
    ftotal = open(os.path.join(sys.argv[1], "total.html"), 'w')
    ftotal.write("<html><style>td {vertical-align:top;}img {max-width:320px;max-height:320}</style>")

    n = min(500, len(simSearch.index))
    for x in range(n):
        print x
        np_name = os.path.join(sys.argv[1], "%d.npy" % x)
        html_name = os.path.join(sys.argv[1], "%d.html" % x)
        jpg_name = os.path.join(sys.argv[1], "%d.jpg" % x)
        img = np.array(np.load(np_name))
        results = simSearch.search(img, 10)        
        draw_html(results, html_name)
        ftotal.write("<a href=%s><img src=%s></a>" % ("%d.html" % x, "%d.jpg" % x))

    ftotal.write("</html>")
    
    
