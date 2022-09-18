function [U,V,iter] = myFCM(X,c,m,V)
% inputs:
% X - [N x d] array of feature vectors
% c - number of clusters
% m - fuzzifier
% outputs:
% U - [c x N] partition matrix
% V - [c x d] cluster center vectors
U = zeros(c,size(X,1));
U = U';
Vold = zeros(c,size(X,2));
Vnew = zeros(c,size(X,2));
iter = 0;
maxiter = 100;
eps = 10^-5;
while (max(sum((V-Vold).^2,2) > eps) && iter < maxiter)
    Vold  = V;
    U = zeros(c,size(X,1));
    U = U';
    for i = 1:size(X,1)
        for j = 1:c
            for k = 1:c
                U(i,j) = U(i,j) + (distance2(X(i,:),V(j,:))^(1/(m-1)))/((distance2(X(i,:),V(k,:)))^(1/(m-1)));
            end
            U(i,j) = U(i,j).^-1;
        end
    end
    U(isnan(U)) = 1;
    for j = 1:c
        Vnew(j,:) = sum(U(:,j).^m .*X,1) / sum(U(:,j).^m);
    end
    V = Vnew;
    iter = iter + 1;
end
end








