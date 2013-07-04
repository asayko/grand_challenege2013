function [dist] = histDist(h1, h2)
    hist1 = histc(h1, 1:16384);
    hist2 = histc(h2, 1:16384);
    dist = histRawDist(hist1, hist2);
end
