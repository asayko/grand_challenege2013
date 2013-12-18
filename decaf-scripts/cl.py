from decaf.scripts.imagenet import DecafNet
import numpy as np
from skimage import io
from skimage import transform
import sys
import glob
import os
import shutil

def load_and_resize(fname):
    print fname
    readed = io.imread(fname)
    print readed.shape
    (w, h, d) = readed.shape
    M = min(w, h)
    frame_size = 227
    if M > frame_size:
        k = 227.0 / M;
        new_shape = (int(k * w), int(k * h))
        resized = transform.resize(readed, new_shape)
    else:
        resized = readed
    return resized, np.asarray(resized)


if __name__ == '__main__':
    folder = sys.argv[1]
    out_folder = sys.argv[2]
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.mkdir(out_folder)
    imgs = glob.glob(os.path.join(folder,"*.jpeg"))
    print imgs
    net = DecafNet('../imagenet_pretrained/imagenet.decafnet.epoch90', '../imagenet_pretrained/imagenet.decafnet.meta')

    flog = open('log.txt', 'w')

    for i, imgname in enumerate(imgs):
        flog.write("%s\t%d" % (imgname, i))
        try:
            resized, img = load_and_resize(imgname)
        except ValueError:
            print "error when read %s" % imgname
            continue
        scores = net.classify(img, center_only=True)
        feature = net.feature('fc6_cudanet_out')
        print feature
        
        out_file = open(os.path.join(out_folder, "%d.npy" % i), 'w')
        np.save(out_file, feature)
        io.imsave(os.path.join(out_folder, "%d.jpg" % i), resized)

