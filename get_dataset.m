%% Function to get the relevant datasets and the associated variables
% Arun Venkitaraman 2018-01-01

%% Outputs
% D: Data matrix with rows as the observations and features/nodes along
% columns
% L: graph-Laplacian
% Ntrain: subset of indices of D corresponding to training set
% Ntest: data indices corresponding to test data
%m: size of graph
%offset: offset of days to be used in the case of temperature data
% city_ip: is the portion of the entrie data used for input (for example
% some cities in the ETEX data)
% city_op: is the portion of the entrie data used for output that lies over associated graph with Laplacian L 



% The input varaible 'dataset, takes values
%'temp' for Swedish temperature data from 2017 for one-day prediction
%'temp2' for Swedish temperature data from 2018 with airpressure input,
%temperature output

% 'etex' for ETEX atmospheric tracer diffusion data
% 'eeg' for EEG data
% 'cere' for fMRI data of the cerebellum region
%'barabasi' for Barabasi-Albert small-world graphs of default size m=10,
% which may be changed below
%'erdos' for Erdos-Renyi small-world graphs of default size m=10,
% which may be changed below


%The descritpiton of the associated datasets may be found in the links mentioned in our article
% The datasets along with the partition o total data into training and test
% set used in the article are prvided alongwith the code.



function  [D,L,alpvec,betvec, Ntrain,Ntest,m,offset,city_ip,city_op]=get_dataset(dataset)
switch dataset
    
    
    
%% Typical data format
% 1)Should have datamatrix D

% 2)The training and testing set of indices Ntrain and Ntest 

% 3)No of nodes m

% 4)city_ip: the subset of nodes used for input

% 5)city_op: subset of nodes used for output

% 6)offset: is the offset between input and output: For example in case of
% one day temperature prediction: We use offset=1, then the input and
% output are given by D(Ntrain-offset,city_ip) and D(Ntrain, city_op),
% respectively. offset is 0 when there is no such relation between input
% and output

% 7)alpvec: is the grid for searching for the first hyperparameter in
% crossvalidation

% 8)betvec: is the grid for searching for the second hyperparameter in
% crossvalidation

% 9) Laplacian matrix has to be provided or constructed as in the case of
% EEG data

    

%% Temperature data
% In case of both 'temp' and 'temp2', we make use of the same 45 cities in
% Sweden given in matrix 'city45data.mat'. However, the users may change
% this number m to less than 45 for 'temp' or choose an appropriate subset

% In case of 'temp2', since airpressure measurements were available at only
% 25 of the 45 cities, m has to <=25 and the nodes/cities must be
% necessarily from the index set set18;


    case 'temp'
    
    load('city45data.mat');
    load('city45T.mat');
    load('smhi_temp_17.mat');
    load('smhi17_partition.mat');
    T=[temp_17'];
    alpvec=[logspace(-3,2,10)];
    betvec=[logspace(-2,2,10)];
    
    
    m=25;
    city_ip=1:m;
    city_op=1:m;
    offset=1;  % The number of days offset between input and output temperatures
    ns=92;
    m=length(city_op);
    
    
    
    A=A45(1:m,1:m);
    A=A.^2;
    A=exp(-A/mean(A(:)));
    A=A-diag(diag(A));
    L=diag(sum(A,2))-A;
    L=L;
    
    
    T=T/max(abs(T(:)));
    D=T';
    
   
%% EEG data
case 'eeg'
    
    load('eegdata_S006R01.mat');
    %load('eegdata_S002R01.mat');
    D=l(1:64,:)';
    alpvec=[logspace(-2,2,10)];
    betvec=[logspace(-2,3,10)];
    %betvec=1;
    La=length(alpvec);
    Lb=length(betvec);
    tic
    %%
    offset=0;
    ntr=5000;    
    % Here i have used the first 5000 samples to construct the graph adjacency matrix based on distance, and used the remaining data for analysis
    
    Y_train=(D(1:ntr,:));
    % Cosntructing the graph
    Afull=pdist2(Y_train',Y_train').^2;
    Afull=Afull/mean(Afull(:));
    Afull=exp(-1*Afull);
    Lfull=diag(sum(Afull,2))-Afull;
    Lfull=Lfull/abs(max(eig(Lfull)));
    
    Afull=Afull-diag(diag(Afull));
    dd=sum(Afull,2);
    [vd,bd]=sort(dd,'descend');
    
    city_ip=1:4:64;
    offset=0;
    city_op=setdiff((1:64),city_ip);
    m=length(city_op);
 
    A=pdist2(Y_train(:,city_op)',Y_train(:,city_op)').^2;
    A=A/mean(A(:));
    A=exp(-1*A);
    A=A.*(A>.25);
    A=A-diag(diag(A));
    L=diag(sum(A,2))-A;
    L=L/abs(max(eig(L)));
    
    m=length(city_op);
    ns=160;
    D=D(ntr:end,:);
    T=D';
    load('EEG_partition.mat')

%% ETEX tracer diffusion data
case 'etex'
    
    
    load('etex_1.mat');
    load('etex_2.mat');
    load('A_etex.mat');
    D=[pmch pmcp]';
    T=D;

    dol=max(abs(D), [], 2);
    D = bsxfun(@rdivide, D, dol);;
    
    alpvec=[logspace(-1,3,10)];
    betvec=[logspace(-1,1,10)];
    
    La=length(alpvec);
    Lb=length(betvec);
    tic
    %%
    %
    m=168;
    
    city_op=1:80;
    city_ip=81:168;
    offset=0;
    ns=60;
    m=length(city_op);
    
    
    A=A(city_op,city_op);
    A=A.^2;
    A=A/mean(A(:));
    A=exp(-A);
    A=A-diag(diag(A));
    L=diag(sum(A,2))-A;
    L=L/abs(max(eig(L)));
    
    D=D(1:end,:);
    T=D';
    T=T/max(abs(T(:)));
    
    Ntrain=randperm(ns,ns/2);
    Ntest=setdiff(1:ns,Ntrain);
    load('etex_partition.mat');
    
%% FMRI data from cerebellum
case'cere'
    
    load('A_cerebellum.mat');
    load('signal_set_cerebellum.mat');
    T=F2;
    
    
    alpvec=[logspace(-2,2,10)];
    betvec=[logspace(-3,2,10)];
 
    La=length(alpvec);
    Lb=length(betvec);
    tic
    
    
    %%
    m=50;
    city_ip=1:10;
    offset=0;
    city_op=setdiff((1:m),city_ip);
    
    
    m=length(city_op);
    A=full(A(city_op,city_op));
    L=diag(sum(A,2))-A;
    ns=290;
    
  
    T=T/max(abs(T(:)));
    D=T';
    %Ntrain=randperm(ns,n);
    %Ntest=setdiff(1:ns,Ntrain);
    load('Cere_partition.mat');
    


    %% Flow ctometry data
    case 'cyto'
    
load('celldatafull.mat');
load('cellA.mat');
D=log10(Xfull(1:7000,:));


%D=normr(D);
%D=D(:,city_op);%
%D=D*diag(1./sqrt(diag(cov(D))));

T=D';
ntr=1;

m=11;
city_ip=[10 11] ;
city_op=setdiff(1:m,city_ip);
offset=0;
ns=1000;
m=length(city_op);

mo=m;
Y_train=(D(1:ntr,city_op));
% 
% Afull=pdist2(Y_train',Y_train').^2;
% Afull=Afull/mean(Afull(:));
% Afull=exp(-1*Afull);
% Afull=Afull-diag(diag(Afull));
% Lfull=diag(sum(Afull,2))-Afull;
% Lfull=Lfull/abs(max(eig(Lfull)));
% L=Lfull;


A=cellA(city_op,city_op);
A=(A+A');
% A=pdist2(Y_train',Y_train').^2;
% A=A/mean(A(:));
% A=exp(-1*A);
% A=A.*(A>.5);
L=diag(sum(A,2))-A;
L=L/max(abs(eig(L)));
alpvec=[logspace(-2,3,10)];
%alpvec=1;

betvec=[logspace(-2,2,10)];
%betvec=1;
La=length(alpvec);
Lb=length(betvec);
tic
%%


D=D(ntr:end,:);
T=D';
T0=T;
T=T/max(abs(T(:)));
%T=normc(T);
Ntrain=1:ns/2;
Ntest=setdiff(1:ns,Ntrain);



end
 
