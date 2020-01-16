function A = Sthresh(A,sp)
% Hard-thresholding operator
% Input:
%         A : coefficients (array)
%         sp: number of non-zero elements (scalar)
%
% Output:
%         A : sparse coefficients (array)
%

N = ndims(A);
% Convert the tensor into a matrix
if N>2
    a = Unfold(A,size(A),N);
else
    a = A;
end
% Keep the highest elements of the coefficients of each sample (row) and zero out others
for i = 1:size(a,1)
    tmp = a(i,:); 
    B = sort(tmp,'descend');
    val = B(sp+1);
    tmp(tmp<=val) = 0;
    a(i,:) = tmp;
end
% Fold the tensor
if N>2
    A = Fold(a,size(A),N);
else
    A = a;
end
end