% --- Tensor Dictionary Learning with representation quantization for Remote Sensing Observation Compression --- %
% --------------------- Anastasia Aidini, Grigorios Tsagkatakis, and Panagiotis Tsakalides --------------------- %

clear
close all
clc

addpath('tensor_toolbox')
rand('seed',2018);

load('Data/training_data.mat') % Training data
load('Data/testing_data.mat')  % Testing data

% Parameters
sp = 0.8;    % sparsity level
K = 500;     % number of atoms of the tensor dictionary (must be K = k*num_train, for some k)
bit = 8;     % number of bits of quantization
p = 0.6;     % step-size parameter
itter = 50;  % number of maximum iterations
tol = 1e-5;  % tolerance

N = ndims(Xtrain)-1;  % number of modes of the tensor data
NwayM = size(Xtrain); % size of the samples
NwayM = NwayM(1:end-1);
d = size(Xtrain,N);           % window size - number of days
num_train = size(Xtrain,N+1); % number of training samples
num_test = size(Xtest,N+1);   % number of testing samples

%% Training process
fprintf('Training \n')

% Initialization of the variables
D = zeros([NwayM K]);   % Tensor Dictionary
R = K/num_train;
for i = 1:num_train
    train = Xtrain(:,:,:,i);
    T = cp_als(tensor(train),R); % CP decomposition of the training samples
    for r = 1:R
        D(:,:,:,(i-1)*R+r) = outprod(T.U{1}(:,r),outprod(T.U{2}(:,r),T.U{3}(:,r)));
    end
end
D = D./max(D(:));
A = zeros(num_train,K); % Sparse Coefficients
G = A;                  % Auxiliary variable for coefficients
Y = zeros(size(G));     % Lagrange multiplier matrix

er_train = []; % training error at each iteration
for i = 1:itter
    X1 = double(ttm(D,G,N+1)); % reconstructed tensor of the previous iteration
    fprintf('Iteration: %d\n',i)
    % Update A
    A = (Unfold(Xtrain,size(Xtrain),N+1)*(Unfold(D,size(D),N+1))'+Y+p*G)*pinv(Unfold(D,size(D),N+1)*(Unfold(D,size(D),N+1))'+p*eye(K,K));
    % Update G
    tmp = A-(1/p)*Y;
    G = Sthresh(tmp,round((1-sp)*K)); % hard-thresholding
    % Update the dictionary D
    D = D+double(ttm(Xtrain,pinv(A),N+1));
    D = D./max(D(:));
    % Update Y
    Y = Y+p*(G-A);
    
    % Stopping criterion
    Xtn = double(ttm(D,G,N+1)); % reconstructed tensor
    er_train(i) = norm(Xtn(:)-Xtrain(:))/norm(Xtrain(:));
    if (er_train(i) <= tol) || ((norm(Xtn(:)-X1(:))/norm(X1(:)))< tol)
        break;
    end
end

%% Learn the symbols and the dictionary for Huffman coding
mn = min(G(:));
mx = max(G(:));
q = (mx-mn)/(2^bit-2);

symbols = [0 mn:q:mx];
symbols = unique(symbols); % symbols of the encoding dictionary
prob = zeros(length(symbols),1);
for j = 1:length(symbols)
    prob(j) = length(find(G(:)==symbols(j)));
end
prob = prob./sum(prob);
dict = huffmandict(symbols,prob); % symbol encoding dictionary

%% Testing Process
fprintf('Testing Process\n')

A = zeros(num_test,K); % Sparse Coefficients
G = A;                 % Auxiliary variable for coefficients
Y = zeros(size(G));    % Lagrange multiplier matrix

Mrec = cell(num_test,1);   % Reconstructed testing samples
error = zeros(num_test,1); % testing error
er_test = [];              % error at each iteration
for i = 1:itter
    X1 = double(ttm(D,G,N+1)); % reconstructed tensor of the previous iteration
    fprintf('Iteration: %d\n',i)
    % Update A
    A = (Unfold(Xtest,size(Xtest),N+1)*(Unfold(D,size(D),N+1))'+Y+p*G)*pinv(Unfold(D,size(D),N+1)*(Unfold(D,size(D),N+1))'+p*eye(K,K));
    % Update G
    tmp = A-(1/p)*Y;
    G = Sthresh(tmp,round((1-sp)*K)); % hard-thresholding
    % Update Y
    Y = Y+p*(G-A);

    % Stopping criterion
    Xtn = double(ttm(D,G,N+1)); % reconstructed tensor
    er_test(i) = norm(Xtn(:)-Xtest(:))/norm(Xtest(:));
    if (er_test(i) <= tol) || ((norm(Xtn(:)-X1(:))/norm(X1(:)))< 1e-3)
        break;
    end
end

%% Quantize the sparse coefficients
bpppb = zeros(num_test,1);         % bits per pixel ber band - bpppb (where band is a time instant)
lam_test = cell(num_test,1);       % coefficients of the testing samples
lam_test_quant = cell(num_test,1); % quantized coeficients of the testing samples

for i = 1:num_test
    fprintf('Testing Sample %d/%d\n',i,num_test)
    test = Xtest(:,:,:,i);
    % Obtain the sparse coefficients
    lam_test{i} = G(i,:)';
    
    % Quantize the sparse coefficients
    lam_test_quant{i} = zeros(length(lam_test{i}),1);
    for j = 1:length(lam_test{i})
        l = find(symbols<lam_test{i}(j));
        u = find(symbols>lam_test{i}(j));
        if isempty(u)
            lam_test_quant{i}(j) = symbols(end);
        elseif isempty(l)
            lam_test_quant{i}(j) = symbols(1);
        else
            lam_test_quant{i}(j) = symbols(l(end)+1);
        end
    end
    
    % Huffman coding
    hcode = huffmanenco(lam_test_quant{i},dict); % encoded measurements - the data we receive
    
    % bpppb
    bits_huffman = numel(hcode);
    bpppb(i) = bits_huffman/prod(size(test));
    
    % Decode the received data
    lam_dec = huffmandeco(hcode,dict);
    
    % Reconstruct the tensor
    Mrec{i} = double(ttm(D,lam_dec',N+1));

    % Compute the NMSE
    error(i) = norm(Mrec{i}(:)-test(:))/norm(test(:));
end

% Plot the error
figure;
plot(error)
xlim([1 num_test])
ylim([0 0.4])
xlabel('Test Samples')
ylabel('NMSE')
