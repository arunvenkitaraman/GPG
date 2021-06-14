% Arun Venkitaraman 2018-06-01

function [all_gam,all_alp,all_mse]=return_GPG_xvalparams(X_train, Y_train,T_train,n,L, R, gamvec,alpvec,sig_ker,SNR)

%% INPUTS
% X_train : input data matrix, samples along rows
% Y_train : output data matrix, samples along rows
% n: no of observation samples
% m : graph size
% L: graph Laplacian
% R: no of partitions for xvalidation
% alpvec: range of alpha values for grid search
% betvec: range of beta values for search

% OUTPUT
%This function returns \alpha and \beta parameters for
%1. GP-L 2. GPG-L 3. GP-K, and 4. GPG-K

m=size(L,2);
La=length(gamvec);
Lb=length(alpvec);
indices=crossvalind('Kfold',n,R);

for r=1:R
    test=(indices==r); % Indices for Xvalidation
    train=~test;
    no=sum(train);
    ltest=sum(test);
    ltrain=sum(train);
    
    Phi_train=X_train(train,:);
    Phi_test=X_train(test,:);
    
    % Internal loops for both alpha and beta parameters
    for b=1:Lb
        for a=1:La
            
            gam=gamvec(a);
            alp=alpvec(b);
            
            %% Test data
            ytrain=T_train(train,:);
            ytrain0=Y_train(train,:);
            ytest=Y_train(test,:);
            
            K1=Phi_train*Phi_train';
            k1=(Phi_train*Phi_test')';
            
            K2=pdist2(Phi_train,Phi_train).^2;
            sig_rbf=sig_ker*mean(K2(:));
            K2=exp(-K2/sig_rbf);
            k2=pdist2(Phi_test,Phi_train).^2;
            k2=exp(-k2/sig_rbf);
            
            y_lin_test=zeros(ltest,m);
            y_lin_g_test=zeros(ltest,m);
            
            y_ker_test=zeros(ltest,m);
            y_ker_g_test=zeros(ltest,m);
            y_lin_train=zeros(ltrain,m);
            y_lin_g_train=zeros(ltrain,m);
            
            y_ker_train=zeros(ltrain,m);
            y_ker_g_train=zeros(ltrain,m);
            
            for nt=1:ltest
                k1n_plus1=(Phi_test(nt,:)*Phi_test(nt,:)')';
                
                k2n_plus1=pdist2(Phi_test(nt,:),Phi_test(nt,:)).^2;
                k2n_plus1=exp(-k2n_plus1/sig_rbf);
                
                % Setting the precisition parameter beta assuming true SNR
                % is known
                bet=inv(((norm(Y_train,'fro')^2/(length(Y_train(:))))*10^(-SNR/10))); 
                
                [mu1,Sig1]=predictiveDistribution(vec(ytrain),zeros(m),K1/gam,k1(nt,:)/gam,k1n_plus1/gam,0,bet,m,no);
                [mu2,Sig2]=predictiveDistribution(vec(ytrain),L,K1/gam,k1(nt,:)/gam,k1n_plus1/gam,alp,bet,m,no);
                [mu3,Sig3]=predictiveDistribution(vec(ytrain),zeros(m),K2/gam,k2(nt,:)/gam,k2n_plus1/gam,0,bet,m,no);
                [mu4,Sig4]=predictiveDistribution(vec(ytrain),L,K2/gam,k2(nt,:)/gam,k2n_plus1/gam,alp,bet,m,no);
                y_lin_test(nt,:)=mu1';
                y_lin_g_test(nt,:)=mu2';
                
                y_ker_test(nt,:)=mu3';
                y_ker_g_test(nt,:)=mu4';
            end
            
            %% Training data
            k1=(Phi_train*Phi_train')';
            k2=pdist2(Phi_train,Phi_train).^2;
            k2=exp(-k2/sig_rbf);
            
            for nt=1:ltrain
                k1n_plus1=(Phi_train(nt,:)*Phi_train(nt,:)')';
                
                k2n_plus1=pdist2(Phi_train(nt,:),Phi_train(nt,:)).^2;
                k2n_plus1=exp(-k2n_plus1/sig_rbf);
                
                
                bet=inv(((norm(Y_train,'fro')^2/(length(Y_train(:))))*10^(-SNR/10)));
                
                [mu1,Sig1]=predictiveDistribution(vec(ytrain),zeros(m),K1/gam,k1(nt,:)/gam,k1n_plus1/gam,0,bet,m,no);
                [mu2,Sig2]=predictiveDistribution(vec(ytrain),L,K1/gam,k1(nt,:)/gam,k1n_plus1/gam,alp,bet,m,no);
                [mu3,Sig3]=predictiveDistribution(vec(ytrain),zeros(m),K2/gam,k2(nt,:)/gam,k2n_plus1/gam,0,bet,m,no);
                [mu4,Sig4]=predictiveDistribution(vec(ytrain),L,K2/gam,k2(nt,:)/gam,k2n_plus1/gam,alp,bet,m,no);
                y_lin_train(nt,:)=mu1';
                y_lin_g_train(nt,:)=mu2';
                
                y_ker_train(nt,:)=mu3';
                y_ker_g_train(nt,:)=mu4';
                
                
            end
            
            
            mse_lin_test(a,b, r)=(norm(ytest(:)-y_lin_test(:),2)^2);
            mse_lin_g_test(a,b, r)=(norm(ytest(:)-y_lin_g_test(:),2)^2);
            
            mse_ker_test(a,b,r)=(norm(ytest(:)-y_ker_test(:),2)^2);
            mse_ker_g_test(a,b,r)=(norm(ytest(:)-y_ker_g_test(:),2)^2);
        end
    end
    
    e_test(r)=norm(ytest(:),2)^2;
    
end


for a=1:La
    for b=1:Lb
        
        
        Mse_lin_test(a,b)=(mean(mse_lin_test(a,b,:))./mean(e_test));
        [a1,b1,val1]=matrix_max_loc(Mse_lin_test);
        
        Mse_lin_g_test(a,b)=(mean(mse_lin_g_test(a,b,:))./mean(e_test));
        [a2,b2,val2]=matrix_max_loc(Mse_lin_g_test);
        
        
        Mse_ker_test(a,b)=(mean(mse_ker_test(a,b,:))./mean(e_test));
        [a3,b3,val3]=matrix_max_loc(Mse_ker_test);
        
        Mse_ker_g_test(a,b)=(mean( mse_ker_g_test(a,b,:))./mean(e_test));
        [a4,b4,val4]=matrix_max_loc(Mse_ker_g_test);
        
        
    end
end
all_mse=[val1,val2,val3,val4];
all_gam=gamvec([a1(1) a2(1) a3(1) a4(1)]);
all_alp=alpvec([b2(1) b4(1)]);