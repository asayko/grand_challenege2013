function [f, index] = hist(image, kdtree, vocab)
    [f, d] = vl_sift(image, 'PeakThresh', 1);
    d = single(d);
    [index, distance] = vl_kdtreequery(kdtree, vocab, d);
end

