
clear;
n = 20;                   
data = 'ALLAML.mat';       
load(data);                      
gt = y;
K = length(unique(gt));
lambda_all = [5,10,20,50,80,100,150,200];  
gamma_all = [5,10,50,100,150];
sigma = 2^4;

for lambda_ind =1:length(lambda_all)
    lambda = lambda_all(lambda_ind);
for gamma_ind = 1:length(gamma_all)
    gamma = gamma_all(gamma_ind);
parfor i = 1:10

[a,b] = size(E);
zz = RandStream('mt19937ar','Seed',i);
RandStream.setGlobalStream(zz);
indx = randperm(b);
EC_end = E(:,indx(1:n));% Base clustering result
M = Gbe(EC_end); 
CA = M*M'./n;
[A] = get_kernel(CA,sigma);
[Z] = solver_AKTEC(A,lambda,gamma,K);
[U,S,V] = svd(Z,'econ');
S = diag(S);
r = sum(S>1e-4*S(1));
U = U(:,1:r);S = S(1:r);
U = U*diag(sqrt(S));
U = normr(U);
L = (U*U').^4;
results_C  = spectral_clustering(L,K);
NMI_S(i)= compute_nmi(results_C,gt);
ARI_S(i)= RandIndex(results_C,gt);
Fscore_S(i)= compute_f(results_C,gt);

end
MELT_mean_nmi(lambda_ind,gamma_ind) = mean(NMI_S);
MELT_mean_ari(lambda_ind,gamma_ind) = mean(ARI_S);
MELT_mean_F1(lambda_ind,gamma_ind) = mean(Fscore_S);

MELT_std_nmi(lambda_ind,gamma_ind) = std(NMI_S);
MELT_std_ari(lambda_ind,gamma_ind) = std(ARI_S);
MELT_std_F1(lambda_ind,gamma_ind) = std(Fscore_S);
end
end

Final_Results(1,1) = max(max(MELT_mean_ari));
Final_Results(1,2) = max(max(MELT_mean_F1));
Final_Results(1,3) = max(max(MELT_mean_nmi))







