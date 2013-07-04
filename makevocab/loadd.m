function [f, d] = loadd(matName);
%dummy function to embed into parfor

t = load(matName);
f = t.f;
d = t.d;

