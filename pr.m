function d = pr(v)
%PR Calculate the participation ratio
d = mean(v).^2 / mean(v.^2);

end

