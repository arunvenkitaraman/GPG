function [sig_ker,gam1,gam2,alp1,alp2]=final_params(Ntrain,D,offset,city_ip,city_op,SNR,Mask,perturb,L,n,R,gamvec,alpvec,sigvec)


%% Cross validation to find parameters
% Finding parameters alp, bet, sigma^2 for each of the four cases as
% applicable: LR, LRG, KR, and KRG by running over 'Run' different  partitions of
% training and testing data

m=size(L,1);


S=length(sigvec); % Grid size for sigma^2 parameter
%sigvec=logspace(0,2,S); % Grid range for $\sigma^2$
%sigvec=linspace(1.e3,1.5e3,S); % For Cere sig=1.58e3
%sigvec=linspace(30,40,S); % For temp17 sig=35
%sigvec=logspace(1,2,S); % For eeg sig=5
%Sig_ker=linspace(5,15,S); % For etex sig=5.5


%% Sigma^2 values found by prior experiments
% sig_ker=1.5e3; % Cere
% sig_ker=25;% temp17
%sig_ker=5; % EEG
%sig_ker=25;%ETEX


All_gam_s=zeros(S,4);
All_alp_s=zeros(S,2);
All_mse_s=zeros(S,4);


Run=1;   % No of iterations to average over different randomized subsets of the training data of size n
for  r=1:Run
    
    ns=length(Ntrain);
    ntrain=Ntrain(randperm(ns,n));

    X_train=(D((ntrain)+offset,city_ip));
    Y_train=(D((ntrain),city_op))*pinv(eye(m)+0*L);
    
    % Generating noisy data
    sig_train=1*sqrt((norm(Y_train,'fro')^2/(length(Y_train(:))))*10^(-SNR/10));  % computing the variance for additive noise of given SNR
    
    %% Use for large perturbations
    if perturb==1
        T_train=Mask.*Y_train;
    end
    %% Use for additive noise
    if perturb==0
        T_train=(Y_train+1*sig_train*randn(size(Y_train))); %
    end
    
    for ss=1:S
        sig_ker=sigvec(ss);
        %%  alpha, beta parameters obtained from the crossvalidation step for a given sigma^2 value
        
        [all_gam_s(ss,:),all_alp_s(ss,:),all_mse_s(ss,:)]=return_GPG_xvalparams(X_train, Y_train,T_train,n,L,R,gamvec,alpvec,sig_ker,SNR);
        
        
    end
    All_gam_s=All_gam_s+all_gam_s;
    All_alp_s=All_alp_s+all_alp_s;
    All_mse_s=All_mse_s+all_mse_s;
end

[~,ii]=min(All_mse_s(:,3)/Run); % Finding the sigma^2 that minimizes the test error for GPG-K

all_gam=All_gam_s(ii,:)/Run;
all_alp=All_alp_s(ii,:)/Run;
sig_ker=sigvec(ii);  % Finding the sig that results in minimum MSE for training set

gam1=all_gam(1); % alpha for GP-L
gam2=all_gam(3);% alpha for GP-K


alp1=all_alp(1);% beta for GPG-L
alp2=all_alp(2);% beta for GPG-K


