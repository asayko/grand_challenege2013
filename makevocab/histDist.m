function [dist] = histDist(h1, h2)
    hist1 = histc(h1, 1:16384);
    hist2 = histc(h2, 1:16384);
    den = sum(hist1 + hist2);
    num = sum(hist2.*(hist1 > 0) + hist1.*(hist2 > 0));
    den
    num
    dist = num / den;
end
