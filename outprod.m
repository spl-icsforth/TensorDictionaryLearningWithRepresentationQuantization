function y = outprod(u,v)
    % Outer product of vectors u and v
    my_ndims = @(x)(isvector(x) + ~isvector(x) * ndims(x));
    v_t = permute(v, circshift(1:(my_ndims(u) + my_ndims(v)), [0, my_ndims(u)]));
    y = bsxfun(@times, u, v_t);
end