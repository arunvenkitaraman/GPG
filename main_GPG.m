%% Kernel Regression for signals over graphs
% Arun Venkitaraman 2018-01-01

close all
clear all
clc
tic;
n=4;  % No of training samples to be used for GPG


R=2; % No of folds for R-fold crossvalidation
SNR=5; % Signal to noise ratio of the additive noise

dataset='cyto';  % Dataset to be chosen

[D,L,gamvec,alpvec, Ntrain,Ntest,m,offset,city_ip,city_op]=get_dataset(dataset);


%gamvec=[logspace(-1,2,5)]; % grid for hyperparameter gamma
%alpvec=[logspace(0,3,5)]; % grid for hyperparameter alpha
sigvec=logspace(2,4,5);  % grid for hyperparameter sigma^2
%sigvec=1.5e3; 
%sigvec=17; 



perturb=0; % To study large perturbations/missing data in training set
%perturb=0; % Corresponds to additive noise of signal-to-noise-ratio given by SNR to training data


% D: Data matrix
%L: graph-Laplacian
%Ntrain: subset of indices of D corresponding to training set
%Ntest: data indices corresponding to test data
%m: size of graph
%offset: offset of days to be used in the case of temperature data
% city_ip: is the portion of the entrie data used for input (for example
% some cities in the ETEX data)
% city_op: is the portion of the entrie data used for output that lies over associated graph with Laplacian L

%% Mask to simulate random large perturbations in training data
Mask=ones(n,m);
for i=1:n
    Mask(i,randperm(m,5))=0; % This value set to 0 simulates missing samples, 1 means no noise, >1 implies large perturbation
end


%% Obtaining sigma^2 (given by sig_ker), alpha and beta for all four cases.
[sig_ker,gam1,gam2,alp1,alp2]=final_params(Ntrain,D,offset,city_ip,city_op,SNR,Mask,perturb,L,n,R,gamvec,alpvec,sigvec);



%% GPG on test data
% Now running the experiment using estimated hyperparameters for many
% different subsets of size n from training data keeping same testing data

for r=1:100
    
    ns=length(Ntrain);
    ntrain=Ntrain(randperm(ns,n));
    ntest=Ntest;
    
    
    ntest=ntest;
    ltrain=length(ntrain);
    ltest=length(ntest);
    X_train=(D((ntrain)+offset,city_ip));
    Y_train=(D((ntrain),city_op))*pinv(eye(m)+0*L);
    X_test=(D((ntest)+offset,city_ip));
    Y_test=(D((ntest),city_op))*pinv(eye(m)+0*L);
    
    % Generating noisy data
    sig_train=1*sqrt((norm(Y_train,'fro')^2/(length(Y_train(:))))*10^(-SNR/10));  % computing the variance for additive noise of given SNR
    
    
    if perturb==1
        % Use for large perturbations
        T_train=Mask.*Y_train;
    end
    
    if perturb==0
        % Use for additive noise
        T_train=(Y_train+1*sig_train*randn(size(Y_train))); %
    end
    
    ytrain0=Y_train(1:n,:);
    ytrain=T_train(1:n,:);
    ytest=Y_test;
    
    
    clear train;
    
    Phi_train=X_train;
    Phi_test=X_test;
    
    ytrain=T_train;
    
    K1=Phi_train*Phi_train';
    k1=(Phi_train*Phi_test')';
    
    K2=pdist2(Phi_train,Phi_train).^2;
    sig_rbf=sig_ker*mean(K2(:));
    K2=exp(-K2/sig_rbf);
    k2=pdist2(Phi_test,Phi_train).^2;
    k2=exp(-k2/sig_rbf);
    

    
    %% For test data
    
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
        
        
        bet=inv(((norm(Y_train,'fro')^2/(length(Y_train(:))))*10^(-SNR/10)));
        
        
        [mu1,Sig1]=predictiveDistribution(vec(ytrain),zeros(m),K1/gam1,k1(nt,:)/gam1,k1n_plus1/gam1,0,bet,m,n);
        [mu2,Sig2]=predictiveDistribution(vec(ytrain),L,K1/gam1,k1(nt,:)/gam1,k1n_plus1/gam1,alp1,bet,m,n);
        [mu3,Sig3]=predictiveDistribution(vec(ytrain),zeros(m),K2/gam2,k2(nt,:)/gam2,k2n_plus1/gam2,0,bet,m,n);
        [mu4,Sig4]=predictiveDistribution(vec(ytrain),L,K2/gam2,k2(nt,:)/gam2,k2n_plus1/gam2,alp2,bet,m,n);
        y_lin_test(nt,:)=mu1';
        y_lin_g_test(nt,:)=mu2';
        
        y_ker_test(nt,:)=mu3';
        y_ker_g_test(nt,:)=mu4';
        
        
        
    end
    
   %% For training data 
    k1=(Phi_train*Phi_train')';
    k2=pdist2(Phi_train,Phi_train).^2;
    k2=exp(-k2/sig_rbf);
    for nt=1:ltrain
        k1n_plus1=(Phi_train(nt,:)*Phi_train(nt,:)')';
        
        k2n_plus1=pdist2(Phi_train(nt,:),Phi_train(nt,:)).^2;
        k2n_plus1=exp(-k2n_plus1/sig_rbf);
        
        
        bet=inv(((norm(Y_train,'fro')^2/(length(Y_train(:))))*10^(-SNR/10)));
        
        
        [mu1,Sig1]=predictiveDistribution(vec(ytrain),zeros(m),K1/gam1,k1(nt,:)/gam1,k1n_plus1/gam1,0,bet,m,n);
        [mu2,Sig2]=predictiveDistribution(vec(ytrain),L,K1/gam1,k1(nt,:)/gam1,k1n_plus1/gam1,alp1,bet,m,n);
        [mu3,Sig3]=predictiveDistribution(vec(ytrain),zeros(m),K2/gam2,k2(nt,:)/gam2,k2n_plus1/gam2,0,bet,m,n);
        [mu4,Sig4]=predictiveDistribution(vec(ytrain),L,K2/gam2,k2(nt,:)/gam2,k2n_plus1/gam2,alp2,bet,m,n);
        y_lin_train(nt,:)=mu1';
        y_lin_g_train(nt,:)=mu2';
        
        y_ker_train(nt,:)=mu3';
        y_ker_g_train(nt,:)=mu4';
        
        
    end
    
    
    % MSE
    mse_lin_train_f(r)=(norm(ytrain0(:)-y_lin_train(:),2)^2);
    mse_lin_g_train_f(r)=(norm(ytrain0(:)-y_lin_g_train(:),2)^2);
    e_train_f(r)=norm(ytrain0(:),2)^2;
    mse_lin_test_f(r)=(norm(ytest(:)-y_lin_test(:),2)^2);
    mse_lin_g_test_f(r)=(norm(ytest(:)-y_lin_g_test(:),2)^2);
    e_test_f(r)=norm(ytest(:),2)^2;
    
    mse_ker_train_f(r)=(norm(ytrain0(:)-y_ker_train(:),2)^2);
    mse_ker_g_train_f(r)=(norm(ytrain0(:)-y_ker_g_train(:),2)^2);
    mse_ker_test_f(r)=(norm(ytest(:)-y_ker_test(:),2)^2);
    mse_ker_g_test_f(r)=(norm(ytest(:)-y_ker_g_test(:),2)^2);
end

%% Final MSE (in dB) values of four approches:
%1. GP-L 2. GPG-L 3. GP-K, and 4. GPG-K
allmse=10*log10([mean(mse_lin_test_f)/mean(e_test_f) mean(mse_lin_g_test_f)/mean(e_test_f) mean(mse_ker_test_f)/mean(e_test_f) mean(mse_ker_g_test_f)/mean(e_test_f) ])


%% Plotting an example realization
figure, plot(ytrain0(:),'k','Linewidth',2), hold on,plot(ytrain(:),'g'), hold on
hold on, plot(y_lin_train(:),'b'),plot(y_lin_g_train(:),'r'),
plot(y_ker_train(:),'bO-'),plot(y_ker_g_train(:),'rO-')

figure, plot(ytest(:),'k','Linewidth',2),  hold on,
plot(y_lin_test(:),'b'),plot(y_lin_g_test(:),'r'),
plot(ytest(:),'k','Linewidth',2),
plot(y_ker_test(:),'bO-'),plot(y_ker_g_test(:),'rO-')
%
