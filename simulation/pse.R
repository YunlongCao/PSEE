rm(list = ls())
library(rootSolve)
library(Rcpp)
library(RcppArmadillo)
library(parallel)
library(foreach)
library(doParallel)
library(glmnet)



n = 1000
q = 10
s = 1
lambda = 0.25
delta = 0.25
#tau = 1
#gamma = 1
alpha = 0.01
fold = 6
maxiter = 40
Lambda.vec = exp(seq(-6,1,0.4))
#Lambda.vec = c(0.03, 0.05)
eps = 10^-6
tol = 10^-4
h = 0.5

beta0 = c(rep(1,s), rep(0,q-s))
kappa = c(rep(1, q/2), rep(-1, q/2))
#kappa = rnorm(q, 0, gamma^2)
beta = c(beta0, 2)

X = matrix(0, n, q)
for (i in 1:q)
{
  X[, i] = rnorm(n, 0, 1)
}
nk = 15
#cc = gaussHermite(num)
#weight = cc$w
#node = cc$x
weight = c(1.522476e-09, 1.059116e-06, 1.000044e-04, 2.778069e-03, 3.078003e-02, 1.584889e-01, 4.120287e-01, 
           5.641003e-01, 4.120287e-01, 1.584889e-01, 3.078003e-02, 2.778069e-03, 1.000044e-04, 1.059116e-06, 1.522476e-09)
node = c(-4.499991e+00, -3.669950e+00, -2.967167e+00, -2.325732e+00, -1.719993e+00, -1.136116e+00, 
         -5.650696e-01, 3.552714e-15, 5.650696e-01, 1.136116e+00, 
         1.719993e+00, 2.325732e+00, 2.967167e+00, 3.669950e+00, 4.499991e+00)

num_cores <- 80
cl <- makeCluster(num_cores)
registerDoParallel(cl)

#iterate_function <- function(k) {
result_list <- foreach(i = 1:1000, .combine = 'rbind', .packages = c('rootSolve', 'Rcpp', 'RcppArmadillo', 'glmnet')) %dopar% 
{
  t1 = Sys.time()
  sourceCpp("pse.cpp")
  
  #U = rbinom(n, 1, 0.2)
  #U = rbeta(n, 2, 2)
  U = rnorm(n, 0, 1)
  EZ = X%*%kappa + lambda*U
  Z = rnorm(n, EZ, 1)
  logit_EY = cbind(X, Z)%*%beta + delta*U
  EY = 1 / (1 + exp(-logit_EY))
  Y = rbinom(n, 1, EY)
  
  S = cbind(X, Z, Y)
  
  #u_vec = c(0, 1)
  #u_prob = c(0.5, 0.5)
  u_vec = seq(-0.5, 0.5, h)
  u_prob = rep(h/(h+1), 1/h+1)
  
  #naive TSLS
  z_x = data.frame(z = Z, data.frame(X))
  zx.coeff = lm(z ~ -1 + ., data = z_x)$coefficients
  Zhat = X%*%zx.coeff
  y_zhat = data.frame(y = Y, z_naive = Zhat)
  beta.naive = glm(y ~ -1 + ., family = binomial(), data = y_zhat)$coefficients[1]

  y_zhat_X = data.frame(y = Y, data.frame(X), z_naive = Z)
  #beta_gamma.naive = glm(y ~ -1 + ., family = binomial(), data = y_zhat_X)$coefficients
  #beta_gamma_kappa.naive = c(beta_gamma.naive, zx.coeff)

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


  #cv_lasso_result <- cv.glmnet(as.matrix(X_lasso), y_lasso, family = "binomial", alpha = 1, intercept = FALSE)
  #best_lambda_lasso <- cv_lasso_result$lambda.min
  #lasso_model <- glmnet(as.matrix(X_lasso), y_lasso, family = "binomial", alpha = 1, lambda = best_lambda_lasso, intercept = FALSE)
  #lasso_coefficients <- coef(lasso_model)[-1]
  #lasso_weights <- 1/(abs(lasso_coefficients)+0.01)

  #cv_lasso_result2 <- cv.glmnet(as.matrix(X_lasso), y_lasso, family = "binomial", alpha = 1, intercept = FALSE, penalty.factor = lasso_weights)
  #best_lambda_lasso2 <- cv_lasso_result2$lambda.min
  #lasso_model2 <- glmnet(as.matrix(X_lasso), y_lasso, family = "binomial", alpha = 1, lambda = best_lambda_lasso, intercept = FALSE, penalty.factor = lasso_weights)
  #lasso_coefficients2 <- coef(lasso_model2)[-1]


  #oracle TSLS
  z_x = data.frame(z = Z, data.frame(X))
  zx.coeff = lm(z ~ -1 + ., data = z_x)$coefficients
  Zhat = X%*%zx.coeff
  #y_z_x = data.frame(y = Y, z_oracle = Zhat, data.frame(X[, 1:s]))
  y_z_x = data.frame(y = Y, z_oracle = Z, data.frame(X[, 1:s]))
  beta.oracle = glm(y ~ -1 + ., family = binomial(), data = y_z_x)$coefficients[1]
  
  write.table(data.frame(beta.naive, beta.oracle), file = "test.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")
  
  cv = CV(beta_gamma_kappa.lasso, q, lambda, delta, u_vec, u_prob, S, eps, tol, maxiter, Lambda.vec, fold)
  Lambda_best = as.numeric(cv[1]) 
  root2 = multiroot(estimating_eq, beta_gamma_kappa.lasso, q = q, lambda = lambda, delta = delta, u_vec = u_vec, u_prob = u_prob, dataset = S)
  root1 = Newton_Raphson(estimating_eq, beta_gamma_kappa.lasso, q, lambda, delta, u_vec, u_prob, S, eps, tol, maxiter, Lambda_best)
  #root1 = Newton_Raphson(estimating_eq, c(beta,kappa), q, lambda, delta, u_vec, u_prob, S, eps, tol, maxiter, 0.05)
  
  param1 = as.vector(unlist(root1[1]))
  beta.pse = param1[q + 1]
  lambda.pse = param1[1:q]
  variance.pse = sandwich_variance(param1, q, lambda, delta, u_vec, u_prob, Lambda_best, eps, S)[q+1,q+1]
  coverage.pse = as.numeric(beta[q+1]>=beta.pse-1.96*sqrt(variance.pse))+as.numeric(beta[q+1]<=beta.pse+1.96*sqrt(variance.pse))-1
  
  param2 = root2$root
  beta.se = param2[q + 1]
  variance.se = variance_se(param2, q, lambda, delta, u_vec, u_prob, S)[q+1,q+1]
  coverage.se = as.numeric(beta[q+1]>=beta.se-1.96*sqrt(variance.se))+as.numeric(beta[q+1]<=beta.se+1.96*sqrt(variance.se))-1
  
  iter = root1[2]
  diff = root1[3]
  conver = as.numeric(diff < 1e-4)
  
  t2 = Sys.time()
  t = t2 - t1
  write.table(data.frame(t), file = "time.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")
  
  write.table(data.frame(beta.naive, beta.oracle, beta.pse, beta.se, variance.pse, variance.se, iter, conver, coverage.pse, coverage.se, Lambda_best), file = "fourest.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")
  write.table(t(as.vector(unlist(cv[2]))), file = "cvect.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")
  write.table(t(lambda.pse), file = "lambda.txt", append = TRUE, sep = ",", col.names = F, qmethod = "double")
  
}
stopImplicitCluster()
stopCluster(cl)
#mclapply(1:100, iterate_function, mc.cores = 3)
