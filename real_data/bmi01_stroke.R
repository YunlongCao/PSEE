rm(list = ls())
library("tidyverse") #the package to delect NA
library(rootSolve)
library(Rcpp)
library(RcppArmadillo)
library(parallel)
library(foreach)
library(doParallel)
library(glmnet)
#library('nloptr')

setwd("/vhome/cyl/ARIC")
ARIC = read.csv("ARIC_pheno.csv")
ARIC = (ARIC %>% drop_na(bmi01, stroke_baseline))
ARIC.male = ARIC
#signal = read.table("sbp.1e-3.txt") 

sample.id.old = as.vector(unlist(read.table("ID2.txt"))) #All the individuals' ID
#sample.id.new = sample.id.old[sample.id.old %in% ARIC[,2]]
sample.id.new.male = sample.id.old[sample.id.old %in% ARIC.male[,2]]

valid.sample.number = length(sample.id.new.male)

computation.id = sample.id.new.male

#genotype.orig = NULL
#for(i in 1:22)
#{
#  path = paste0("chr", i, ".DS.txt")
#  genotype.orig = cbind(genotype.orig, t(read.table(file = path)))
#} 
genotype.orig = t(read.table(file = '/vhome/cyl/pse/realdata/ARIC_data/DS.txt'))


geno.name = genotype.orig[1,]                              #name of snps
row = dim(genotype.orig)[1]                             #number of snps
genotype = data.frame(id = c(sample.id.old), geno.name = genotype.orig[2:row,])                                  # integrate samples and snps 
names(genotype) = c("id", geno.name)# modify its name
genotype = genotype[genotype[,1] %in% sample.id.new.male,]


genotype.computation = genotype[genotype[,1] %in% computation.id,]


ARIC.computation = ARIC[ARIC[,2]%in% computation.id, ]

pheno_x.computation = ARIC.computation[, colnames(ARIC.computation) %in% c("bmi01")]


pheno_x.inveranktrans.computation = qnorm((rank(pheno_x.computation, na.last="keep")-0.5)/sum(!is.na(pheno_x.computation)))
pheno_y.computation = ARIC.computation[, colnames(ARIC.computation) %in% c("stroke_baseline")]  #phenotype y


#tmp1 = as.data.frame(lapply(genotype.computation[,2:dim(genotype.computation)[2]], as.numeric))
tmp1 = as.data.frame(genotype.computation[,2:dim(genotype.computation)[2]])
tmp1 = apply(tmp1, 2, function(x) as.numeric(as.character(x)))
tmp2 = t(apply(tmp1, 1, function(x) x-apply(tmp1, 2, mean)))
#genotype.center.computation = cbind(genotype.computation[,1], tmp1 - apply(tmp1, 2, mean)) #centering gene matrix
genotype.center.computation = cbind(genotype.computation[,1], tmp2)
names(genotype.center.computation) = names(genotype)

valid.geno.name = names(genotype[2:length(names(genotype))])
#IV.name = intersect(valid.geno.name, signal[,1])
#IV.name = c("9:81349608:G:T", "16:17271327:T:C", "7:131453665:C:G")
#IV.name = c("9:81349608:G:T", "9:104555177:A:C", "16:87085366:G:A", "16:8297685:C:T", "1:54696743:A:G", "1:55504650:G:A")
IV.name = c("9:81349608:G:T", "9:87275895:A:G", "11:47650993:C:T", "9:111932342:C:T", "7:92244422:C:T", "7:121964349:C:T")





Z = pheno_x.inveranktrans.computation 
Y = pheno_y.computation
#X = genotype.center.computation[, IV.name] 
#rownames(X) <- NULL  
#colnames(X) <- NULL  
X = matrix(0, length(Y), length(IV.name))
for (i in 1:length(IV.name))
{
  X[, i] = as.vector(genotype.center.computation[, IV.name[i]])
}

n = length(pheno_x.inveranktrans.computation)
q = length(IV.name)
lambda = 1
delta = 1
#tau = 1
#gamma = 1
alpha = 0.01
fold = 10
maxiter = 40
Lambda.vec = exp(seq(-8,-4,0.2))
#Lambda.vec = exp(seq(-8,1,0.4))
#Lambda.vec = c(0.03, 0.05)
eps = 10^-6
tol = 10^-4
h = 1
nk = 15
#cc = gaussHermite(num)
#weight = cc$w
#node = cc$x
weight = c(1.522476e-09, 1.059116e-06, 1.000044e-04, 2.778069e-03, 3.078003e-02, 1.584889e-01, 4.120287e-01, 
           5.641003e-01, 4.120287e-01, 1.584889e-01, 3.078003e-02, 2.778069e-03, 1.000044e-04, 1.059116e-06, 1.522476e-09)
node = c(-4.499991e+00, -3.669950e+00, -2.967167e+00, -2.325732e+00, -1.719993e+00, -1.136116e+00, 
         -5.650696e-01, 3.552714e-15, 5.650696e-01, 1.136116e+00, 
         1.719993e+00, 2.325732e+00, 2.967167e+00, 3.669950e+00, 4.499991e+00)


#setwd('/vhome/cyl/pse/realdata/hypert05_sbp21')
#sourceCpp("pse.cpp")

S = cbind(X, Z, Y)
S = as.matrix(S)
#u_vec = c(0, 1)
#u_prob = c(0.5, 0.5)
u_vec = seq(0, 1, h)
u_prob = rep(h/(h+1), 1/h+1)

#naive TSLS
z_x = data.frame(z = Z, data.frame(X))
zx.coeff = lm(z ~ -1 + ., data = z_x)$coefficients
Zhat = X%*%zx.coeff
y_zhat = data.frame(y = Y, z_naive = Zhat)
beta.naive = glm(y ~ -1 + ., family = binomial(), data = y_zhat)$coefficients[1]
 
y_zhat_X = data.frame(y = Y, data.frame(X), z_naive = Z)
beta_gamma.naive = glm(y ~ -1 + ., family = binomial(), data = y_zhat_X)$coefficients
beta_gamma_kappa.naive = c(beta_gamma.naive, zx.coeff)

#adaptive lasso estimator for beta_gamma, as initial estimator
y_lasso = y_zhat_X$y
X_lasso = y_zhat_X[, -1]
y_lasso = as.factor(y_lasso)
cv_ridge_result <- cv.glmnet(as.matrix(X_lasso), y_lasso, family = "binomial", alpha = 0.5, intercept = FALSE)
best_lambda_ridge <- cv_ridge_result$lambda.min
ridge_model <- glmnet(as.matrix(X_lasso), y_lasso, family = "binomial", alpha = 0.5, lambda = best_lambda_ridge, intercept = FALSE)
ridge_coefficients <- coef(ridge_model)[-1]
lasso_weights <- 1/abs(ridge_coefficients)
cv_lasso_result <- cv.glmnet(as.matrix(X_lasso), y_lasso, family = "binomial", alpha = 1, intercept = FALSE, penalty.factor = lasso_weights)
best_lambda_lasso <- cv_lasso_result$lambda.min
lasso_model <- glmnet(as.matrix(X_lasso), y_lasso, family = "binomial", alpha = 1, lambda = best_lambda_lasso, intercept = FALSE, penalty.factor = lasso_weights)
adaptive_lasso_coefficients <- coef(lasso_model)
beta_gamma.lasso = adaptive_lasso_coefficients[-1]
beta_gamma.lasso[1:q][beta_gamma.lasso[1:q]<mean(abs(beta_gamma.lasso[1:q]))] = 0
beta_gamma_kappa.lasso = c(beta_gamma.lasso, zx.coeff)

num_cores <- 6
cl <- makeCluster(num_cores)
registerDoParallel(cl)

#iterate_function <- function(k) {
result_list <- foreach(i = 1:length(Lambda.vec), .combine = 'rbind', .packages = c('rootSolve', 'Rcpp', 'RcppArmadillo', 'glmnet')) %dopar% 
{
  #t1 = Sys.time()
  setwd('/vhome/cyl/pse/realdata/bmi01_stroke_v5_42')
  sourceCpp("pse.cpp")
  
  #write.table(data.frame(beta.naive, beta.oracle), file = "test.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")
  
  cv = CV(beta_gamma_kappa.lasso, q, lambda, delta, u_vec, u_prob, S, eps, tol, maxiter, Lambda.vec[i], fold)
  write.table(t(as.vector(unlist(cv[2]))), file = "cvect.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")

}
stopImplicitCluster()
stopCluster(cl) 

setwd('/vhome/cyl/pse/realdata/bmi01_stroke_v5_42')
sourceCpp("pse.cpp")
cvect = read.table("cvect.txt", sep = ",")
#cvect = read_csv("cvect.txt", col_names = FALSE)

Lambda_best = Lambda.vec[which.min(cvect[, 2])]
root2 = multiroot(estimating_eq, beta_gamma_kappa.lasso, q = q, lambda = lambda, delta = delta, u_vec = u_vec, u_prob = u_prob, dataset = S)
root1 = Newton_Raphson(estimating_eq, beta_gamma_kappa.naive, q, lambda, delta, u_vec, u_prob, S, eps, tol, maxiter, Lambda_best)
#root1 = Newton_Raphson(estimating_eq, c(beta,kappa), q, lambda, delta, u_vec, u_prob, S, eps, tol, maxiter, 0.05)
#root1 = Newton_Raphson(estimating_eq, beta_gamma_kappa.naive, q, lambda, delta, u_vec, u_prob, S, eps, tol, maxiter, exp(-8))
param1 = as.vector(unlist(root1[1]))
beta.pse = param1[q + 1]
lambda.pse = param1[1:q]
variance.pse = sandwich_variance(param1, q, lambda, delta, u_vec, u_prob, Lambda_best, eps, S)[q+1,q+1]

param2 = root2$root
beta.se = param2[q + 1]
variance.se = variance_se(param2, q, lambda, delta, u_vec, u_prob, S)[q+1,q+1]

iter = root1[2]
diff = root1[3]
conver = as.numeric(diff < 1e-4)

#t2 = Sys.time()
#t = t2 - t1
#write.table(data.frame(t), file = "time.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")

write.table(data.frame(beta.naive, beta.pse, beta.se, variance.pse, variance.se, iter, conver, Lambda_best), file = "fourest.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")
#write.table(t(as.vector(unlist(cv[2]))), file = "cvect.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")
write.table(t(lambda.pse), file = "lambda.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")


#mclapply(1:100, iterate_function, mc.cores = 3)
