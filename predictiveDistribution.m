function [mu,Sig]=predictiveDistribution(t,L,K,k,kn_plus1,alp,bet,m,n)
% computes and returns the mean and covariance of the one step predictive
% distribution

k=k';
B=inv(eye(m)+alp*L);

D=kron(B*B,k);
F=kn_plus1*(B*B)+inv(bet)*eye(m);

C=kron(B*B,K)+inv(bet)*eye(m*n);
%invC=pinv(C);

% Using Woodbur formula for inverse of matrix
Binv=(eye(m)+alp*L);
Kinv=pinv(K);

A=kron(B*B,K);
Ainv=kron(Binv^2,Kinv);

invC=Ainv-Ainv*pinv(eye(m*n)+inv(bet)*Ainv)*inv(bet)*Ainv;


mu=D'*invC*t;
mu=t'*invC*D;
Sig=F-D'*invC*D;
