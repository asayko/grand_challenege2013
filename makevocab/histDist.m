function [dist] = histDist(h1, h2, vocabSize)
    hist1 = histc(h1, 1:vocabSize);
    hist2 = histc(h2, 1:vocabSize);
    dist = histRawDist(hist1, hist2);
end
