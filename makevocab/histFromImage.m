function [f, index] = histFromImage(image, kdtree, vocab)
    [f, d] = vl_sift(image, 'PeakThresh', 1);
    %size(d)
    %d = single(d);
    %[index, distance] = vl_kdtreequery(kdtree, vocab, d);
    index =  vl_ikmeanspush(d,int32(vocab));
end

