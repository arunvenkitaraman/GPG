% Arun Venkitaraman 2018-01-01

function [x,y,num]=matrix_max_loc(A)

%This function returns the row x and column y where the smallest entry of A
%is located. The value of the entry is returned in 'num
[num] = min(A(:));
[x y] = ind2sub(size(A),find(A==num));

end