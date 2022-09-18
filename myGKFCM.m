function [U,V,S] = myGKFCM(X,c,m,p,V)
% inputs:
% X - [N x d] array of feature vectors
% c - number of clusters
% m - fuzzifier
% p - [c x 1] scale parameter
% outputs:
% U - [c x N] partition matrix
% V - [c x d] cluster center vectors
% S - [N x N x c] covariance matrices

d = size(X,2);
U = zeros(c,size(X,1));
U = U';

S = zeros(size(X,2),size(X,2),c);
Vold = zeros(c,size(X,2));
Vnew = zeros(c,size(X,2));
iter = 0;
maxiter = 100;
eps = 10^-5;

for kk = 1:c
    A(:,:,kk) = eye(d);
end

while (max(sum((V-Vold).^2,2) > eps) && iter < maxiter)
    Vold  = V;
    U = zeros(c,size(X,1));
    U = U';
    for ii = 1:size(X,1)
        for j = 1:c
            for k = 1:c
                U(ii,j) = U(ii,j) + (distance2(X(ii,:),V(j,:),A(:,:,j))^(1/(m-1)))/((distance2(X(ii,:),V(k,:),A(:,:,k)))^(1/(m-1)));
            end
            U(ii,j) = U(ii,j).^-1;
        end
    end
    U(isnan(U)) = 1;
    for j = 1:c
        Vnew(j,:) = sum(U(:,j).^m .*X,1) / sum(U(:,j).^m);
    end
    
    for j = 1:c
        for ii  = 1:size(X,1) 
            S(:,:,j) = S(:,:,j)+ (U(ii,j)^m)*(X(ii,:)-V(j,:))'*(X(ii,:)-V(j,:));
        end
        A(:,:,j) = (p(j)*det(S(:,:,j)))^(1/d)*inv(S(:,:,j));
    end
    V = Vnew;
    iter = iter + 1;
end
end