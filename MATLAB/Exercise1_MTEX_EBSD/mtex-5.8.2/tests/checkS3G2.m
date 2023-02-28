cs = crystalSymmetry('432')
%ss = specimenSymmetry('222')

S3G = equispacedSO3Grid(cs,'resolution',2.5*degree);

ori = orientation.rand(1000);

d = zeros(size(ori));
for k = 1:length(ori)
  progress(k,length(ori));
  d(k) = sum(S3G.find(ori(k),20*degree));

end

%%

histogram(d)