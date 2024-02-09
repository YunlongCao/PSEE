#include <RcppArmadillo.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


int n = 1000;

int nk = 15; 

int q = 10;

double alpha = 0.01;

double eps = 10^-6;

double tol = 10^-4;

int maxiter = 40;

int fold = 6;

double a_scad = 3.7;

std::vector<double> weight_values = {
  1.522476e-09, 1.059116e-06, 1.000044e-04, 2.778069e-03, 3.078003e-02,
  1.584889e-01, 4.120287e-01, 5.641003e-01, 4.120287e-01, 1.584889e-01,
  3.078003e-02, 2.778069e-03, 1.000044e-04, 1.059116e-06, 1.522476e-09
};
arma::vec weight(weight_values);

std::vector<double> node_values = {
  -4.499991e+00, -3.669950e+00, -2.967167e+00, -2.325732e+00, -1.719993e+00, -1.136116e+00, 
  -5.650696e-01, 3.552714e-15, 5.650696e-01, 1.136116e+00, 
  1.719993e+00, 2.325732e+00, 2.967167e+00, 3.669950e+00, 4.499991e+00
};
arma::vec node(node_values);


// [[Rcpp::export]]
double expit(double x) {
  return 1 / (1 + exp(-x));
}

// [[Rcpp::export]]
double p_z(double z, arma::vec x, double u, arma::vec kappa, double lambda) {
  double sum_kappa_x = sum(x % kappa);
  return 1 / sqrt(2 * M_PI) * exp(-0.5 * (z - (sum_kappa_x + lambda * u)) * (z - (sum_kappa_x + lambda * u)));
}


// [[Rcpp::export]]
double p_y(int y, arma::vec x, double z, double u, arma::vec beta, double delta) {
  arma::vec z_vec = arma::vec(1).fill(z);
  arma::vec combine = arma::join_cols(x, z_vec);
  double linear_combination = sum(combine % beta) + delta * u;
  
  if (y == 1)
    return expit(linear_combination);
  else
    return 1 - expit(linear_combination);
}

// Define func
// [[Rcpp::export]]
double func(int y, arma::vec beta, arma::vec kappa, double lambda, double delta,
            double u_j, double u_j_prob, arma::vec X_i, double z,
            arma::vec u_vec, arma::vec u_prob) {
  int n_u_vec = u_vec.size();
  double num = p_y(y, X_i, z, u_j, beta, delta) * u_j_prob;
  double den = 0.0;
  
  for (int i = 0; i < n_u_vec; i++) {
    double exponent = (u_j - u_vec[i]) * (0.5 * lambda * lambda * (u_j + u_vec[i]) - lambda * (z - sum(kappa % X_i)));
    den += exp(exponent) * p_y(y, X_i, z, u_vec[i], beta, delta) * u_prob[i];
  }
  
  if (R_IsNaN(num / den)) {
    den = 0.0;
    for (int i = 0; i < n_u_vec; i++) {
      double exponent = (u_j - u_vec[i]) * (0.5 * lambda * lambda * (u_j + u_vec[i]) - lambda * (z - sum(kappa % X_i)) + (1 - 2 * y) * delta);
      den += exp(exponent) * u_prob[i];
    }
    return u_j_prob / den;
  } else {
    return num / den;
  }
}

// Define compute_I_ij
// [[Rcpp::export]]
double compute_I_ij(arma::vec beta, arma::vec kappa, double lambda, double delta, double u_i, double u_j, double u_j_prob, arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  double sum = 0;
  double a = 0;
  int n_X_i = X_i.n_elem;
  for (int i = 0; i < n_X_i; ++i) {
    a += kappa[i] * X_i[i];
  }
  a += lambda * u_i;
  
  for (int y = 0; y <= 1; ++y) {
    for (int k = 0; k < nk; ++k) {
      double z_k = sqrt(2) * node[k] + a;
      double integrand = func(y, beta, kappa, lambda, delta, u_j, u_j_prob, X_i, z_k, u_vec, u_prob);
      sum += integrand * p_y(y, X_i, z_k, u_i, beta, delta) * weight[k] / sqrt(M_PI);
    }
  }
  return sum;
}

// [[Rcpp::export]]
arma::vec score(arma::vec beta, arma::vec kappa, double lambda, double delta, arma::vec x, double u, double z, int y) {
  arma::vec combined(q + 1); 
  combined.subvec(0, q - 1) = x;
  combined[q] = z; // 
  arma::vec score_beta = (y == 1) * combined - expit(sum(combined % beta) + delta * u) * combined;
  arma::vec score_kappa = (z - (sum(x % kappa) + lambda * u)) * x;
  
  arma::vec score(2 * q + 1); 
  score.subvec(0, q) = score_beta;
  score.subvec(q + 1, 2 * q) = score_kappa;
  return score;
}

// [[Rcpp::export]]
arma::vec score2(arma::vec param, double lambda, double delta, arma::vec x, double u, double z, int y) {
  arma::vec beta = param.subvec(0, q);
  arma::vec kappa = param.subvec(q +1, 2 * q);
  arma::vec combined(q + 1); 
  combined.subvec(0, q - 1) = x;
  combined[q] = z; // 
  arma::vec score_beta = (y == 1) * combined - expit(sum(combined % beta) + delta * u) * combined;
  arma::vec score_kappa = (z - (sum(x % kappa) + lambda * u)) * x;
  
  arma::vec score(2 * q + 1); 
  score.subvec(0, q) = score_beta;
  score.subvec(q + 1, 2 * q) = score_kappa;
  return score;
}

// [[Rcpp::export]]
arma::vec func_2(int y, arma::vec beta, arma::vec kappa, double lambda, double delta, double u_i, arma::vec X_i, double z, arma::vec u_vec, arma::vec u_prob) {
  arma::vec num(2 * q + 1);
  int n_u_vec = u_vec.n_elem;
  for (int i = 0; i < n_u_vec; i++) {
    arma::vec sc = score(beta, kappa, lambda, delta, X_i, u_vec[i], z, y);
    num += sc * func(y, beta, kappa, lambda, delta, u_vec[i], u_prob[i], X_i, z, u_vec, u_prob);
  }
  return num;
}

// [[Rcpp::export]]
arma::vec compute_b_i(arma::vec beta, arma::vec kappa, double lambda, double delta, double u_i, arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  arma::vec summation(2 * q + 1);
  double a = sum(X_i % kappa) + lambda * u_i;
  
  for (int y = 0; y <= 1; y++) {
    for (int k = 0; k < nk; k++) {
      double z_k = sqrt(2) * node[k] + a;
      arma::vec integrand = func_2(y, beta, kappa, lambda, delta, u_i, X_i, z_k, u_vec, u_prob);
      summation += integrand * p_y(y, X_i, z_k, u_i, beta, delta) * weight[k] / sqrt(M_PI);
    }
  }
  return summation;
}

// [[Rcpp::export]]
arma::mat obtain_I(arma::vec beta, arma::vec kappa, double lambda, double delta,
                   arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  int u_len = u_vec.n_elem;
  arma::mat I(u_len, u_len);
  
  for (int i = 0; i < u_len; i++) {
    for (int j = 0; j < u_len; j++) {
      I(i, j) = compute_I_ij(beta, kappa, lambda, delta, u_vec[i], u_vec[j], u_prob[j], X_i, u_vec, u_prob);
    }
  }
  return I;
}

// Define the obtain_b function
// [[Rcpp::export]]
arma::mat obtain_b(arma::vec beta, arma::vec kappa, double lambda, double delta,
                   arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  int u_len = u_vec.n_elem; 
  arma::mat b(u_len, 2 * q + 1); 
  
  for (int i = 0; i < u_len; i++) {
    arma::vec b_i = compute_b_i(beta, kappa, lambda, delta, u_vec[i], X_i, u_vec, u_prob);
    b.row(i) = b_i.t();
  }
  return b;
}

// [[Rcpp::export]]
arma::mat obtain_b2(arma::vec param, double lambda, double delta,
                    arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  arma::vec beta = param.subvec(0, q);
  arma::vec kappa = param.subvec(q +1, 2 * q);
  int u_len = u_vec.n_elem; 
  arma::mat b(u_len, 2 * q + 1); 
  
  for (int i = 0; i < u_len; i++) {
    arma::vec b_i = compute_b_i(beta, kappa, lambda, delta, u_vec[i], X_i, u_vec, u_prob);
    b.row(i) = b_i.t();
  }
  return b;
}

// Define the solve_a function
// [[Rcpp::export]]
arma::mat solve_a(arma::vec beta, arma::vec kappa, double lambda, double delta,
                  arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  int k = u_vec.n_elem; 
  
  // Calculate I and b
  arma::mat I = obtain_I(beta, kappa, lambda, delta, X_i, u_vec, u_prob);
  arma::mat b = obtain_b(beta, kappa, lambda, delta, X_i, u_vec, u_prob);
  
  // Set up the weight matrix W
  arma::mat W(k, k);
  W(0, 0) = 0.5;
  W(k - 1, k - 1) = 0.5;
  for (int i = 1; i < k - 1; i++) {
    W(i, i) = 1.0;
  }
  
  //double h = 1.0 / (k - 1);
  //arma::mat design_matrix = h * I * W;
  arma::mat design_matrix = I;
  
  // Calculate the result
  arma::mat temp = trans(design_matrix) * design_matrix  + alpha * eye(k, k);
  arma::mat result = inv(temp) * trans(design_matrix) * b;
  return result;
}

// [[Rcpp::export]]
arma::vec efficient_score(arma::vec beta, arma::vec kappa, double lambda, double delta, int Y, double Z, arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  arma::mat a = solve_a(beta, kappa, lambda, delta, X_i, u_vec, u_prob);
  
  int k = u_vec.n_elem;
  arma::vec num(2 * q + 1);
  double den = 0;
  
  for (int i = 0; i < k; i++) {
    arma::vec score_i = score(beta, kappa, lambda, delta, X_i, u_vec(i), Z, Y);
    arma::vec a_i = trans(a.row(i));
    arma::vec temp_num = (score_i - a_i) * u_prob(i) * p_z(Z, X_i, u_vec(i), kappa, lambda) * p_y(Y, X_i, Z, u_vec(i), beta, delta);
    num += temp_num;
    den += u_prob(i) * p_z(Z, X_i, u_vec(i), kappa, lambda) * p_y(Y, X_i, Z, u_vec(i), beta, delta);
  }
  
  return num / den;
}

// [[Rcpp::export]]
arma::vec efficient_score2(arma::vec param, double lambda, double delta, int Y, double Z, arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  arma::vec beta = param.subvec(0, q);
  arma::vec kappa = param.subvec(q +1, 2 * q);
  
  arma::mat a = solve_a(beta, kappa, lambda, delta, X_i, u_vec, u_prob);
  
  int k = u_vec.n_elem;
  arma::vec num(2 * q + 1);
  double den = 0;
  
  for (int i = 0; i < k; i++) {
    arma::vec score_i = score(beta, kappa, lambda, delta, X_i, u_vec(i), Z, Y);
    arma::vec a_i = trans(a.row(i));
    arma::vec temp_num = (score_i - a_i) * u_prob(i) * p_z(Z, X_i, u_vec(i), kappa, lambda) * p_y(Y, X_i, Z, u_vec(i), beta, delta);
    num += temp_num;
    den += u_prob(i) * p_z(Z, X_i, u_vec(i), kappa, lambda) * p_y(Y, X_i, Z, u_vec(i), beta, delta);
  }
  
  return num / den;
}

// [[Rcpp::export]]
arma::vec estimating_eq(arma::vec param, int q, double lambda, double delta, arma::vec u_vec, arma::vec u_prob, arma::mat dataset) {
  int num_row = dataset.n_rows;
  arma::vec EE(2 * q + 1);
  
  for (int i = 0; i < num_row; i++) {
    arma::vec beta = param.subvec(0, q);
    arma::vec kappa = param.subvec(q + 1, 2 * q);
    int Y = dataset(i, q + 1);
    double Z = dataset(i, q);
    arma::vec X_i = trans(dataset.row(i).subvec(0, q - 1));
    arma::vec e_s = efficient_score(beta, kappa, lambda, delta, Y, Z, X_i, u_vec, u_prob);
    EE += e_s;
  }
  return EE;
}

// [[Rcpp::export]]
arma::vec q_scad(arma::vec theta, double Lambda, double a = a_scad) {
  int p = theta.n_elem;
  arma::vec b1(p, fill::zeros);
  arma::vec b2(p, fill::zeros);
  
  for (int i = 0; i < p; i++) {
    theta(i) = std::abs(theta(i));
    if (theta(i) > Lambda)
      b1(i) = 1;
    if (theta(i) < (Lambda * a))
      b2(i) = 1;
  }
  
  return Lambda * (1 - b1) + ((Lambda * a - theta) % b2) / (a - 1) % b1;
}

// Returns an approximation of the Jacobian matrix
// [[Rcpp::export]]
arma::mat computeJacobian(Function func, arma::vec x, int q, double lambda, double delta, arma::vec u_vec, arma::vec u_prob, arma::mat dataset) {
  int n_x = 2 * q + 1; 
  
  // Initialize the Jacobian matrix with zeros
  arma::mat jacobian(n_x, n_x);
  
  // Compute the Jacobian matrix numerically
  double epsilon = 1e-6;
  for (int i = 0; i < n_x; ++i) {
    // Create copies of x with perturbation
    arma::vec x_plus = x;
    x_plus(i) += epsilon;
    
    // Compute the corresponding function values
    arma::vec y_plus = as<arma::vec>(func(x_plus, q, lambda, delta, u_vec, u_prob, dataset));
    arma::vec y = as<arma::vec>(func(x, q, lambda, delta, u_vec, u_prob, dataset));
    
    // Compute the gradient and store it in the Jacobian matrix
    arma::vec gradient = (y_plus - y) / epsilon;
    jacobian.col(i) = gradient;
  }
  
  return jacobian;
}

// [[Rcpp::export]]
arma::mat computeJacobian2(Function func, arma::vec x, double lambda, double delta, int Y, double Z, arma::vec X_i, arma::vec u_vec, arma::vec u_prob) {
  int n_x = 2 * q + 1; 
  
  // Initialize the Jacobian matrix with zeros
  arma::mat jacobian(n_x, n_x);
  arma::vec y = as<arma::vec>(func(x, lambda, delta, Y, Z, X_i, u_vec, u_prob));
  
  // Compute the Jacobian matrix numerically
  double epsilon = 1e-6;
  for (int i = 0; i < n_x; ++i) {
    // Create copies of x with perturbation
    arma::vec x_plus = x;
    x_plus(i) += epsilon;
    
    // Compute the corresponding function values
    arma::vec y_plus = as<arma::vec>(func(x_plus, lambda, delta, Y, Z, X_i, u_vec, u_prob));
    
    // Compute the gradient and store it in the Jacobian matrix
    arma::vec gradient = (y_plus - y) / epsilon;
    jacobian.col(i) = gradient;
  }
  return jacobian;
}

// [[Rcpp::export]]
List U_H_E(Function estimating_eq, arma::vec param_new, int q, double lambda, double delta,
           arma::vec u_vec, arma::vec u_prob, arma::mat dataset, double eps, double Lambda) {
  
  arma::vec beta0_new = param_new.subvec(0, q - 1);
  
  
  arma::mat E = arma::zeros<arma::mat>(2 * q + 1, 2 * q + 1);
  E.submat(0, 0, q - 1, q - 1) = diagmat(q_scad(abs(beta0_new), Lambda) / (abs(beta0_new) + eps));
  
 
  arma::mat H = -computeJacobian(estimating_eq, param_new, q = q, lambda = lambda, delta = delta, u_vec = u_vec, u_prob = u_prob, dataset = dataset);
  
  
  arma::vec U = as<arma::vec>(estimating_eq(param_new, q, lambda, delta, u_vec, u_prob, dataset));
  
 
  return List::create(Named("U", U), Named("H", H), Named("E", E));
}

// [[Rcpp::export]]
List Newton_Raphson(Function estimating_eq, arma::vec param, int q, double lambda,
                    double delta, arma::vec u_vec, arma::vec u_prob, arma::mat dataset,
                    double eps, double tol, int maxiter, double Lambda) {
  
  int n_data = dataset.n_rows; 
  
  arma::vec param_new = param;
  arma::vec U;
  arma::mat H, E;
  double diff = 1.0;
  int iter = 0;
  
  while (iter < maxiter) {
    arma::vec param_old = param_new;
    
  
    List U_H_E_val = U_H_E(estimating_eq, param_new, q, lambda, delta, u_vec, u_prob, dataset, eps, Lambda);
    U = as<arma::vec>(U_H_E_val["U"]);
    H = as<arma::mat>(U_H_E_val["H"]);
    E = as<arma::mat>(U_H_E_val["E"]);
    
  
    param_new = param_old + inv(H + n_data * E) * (U - n_data * E * param_old);
    
    
    diff = sum(abs(param_old - param_new));
    iter++;
    
    Rcout << "Iteration: " << iter << std::endl;
    Rcout << "Parameter vector: " << param_new.t() << std::endl;
    Rcout << "Difference: " << diff << ", Tolerance: " << tol << std::endl;
    
    if (diff <= tol)
      break;
  }
  
  return List::create(Named("param_new", param_new), Named("iterations", iter), Named("diff", diff));
}

// [[Rcpp::export]]
arma::mat sandwich_variance(arma::vec param, int q, double lambda, double delta, arma::vec u_vec, arma::vec u_prob, double Lambda, double eps, arma::mat dataset) {
  arma::vec beta = param.subvec(0, q);
  arma::vec beta0 = param.subvec(0, q - 1);
  arma::vec kappa = param.subvec(q + 1, 2 * q);
  int m = dataset.n_rows;
  
  arma::mat M = arma::zeros<arma::mat>(2 * q + 1, 2 * q + 1);
  
  for (int i = 0; i < m; i++) {
    arma::vec X_i = dataset.row(i).subvec(0, q - 1).t();
    double Z = dataset(i, q);
    int Y = dataset(i, q + 1);
    
    arma::vec phi_i = efficient_score(beta, kappa, lambda, delta, Y, Z, X_i, u_vec, u_prob);
    arma::mat M_i = phi_i * phi_i.t();
    
    M += M_i;
  }
  
  
  Function estimating_eq("estimating_eq");
  arma::mat H = -computeJacobian(estimating_eq, param, q, lambda, delta, u_vec, u_prob, dataset);
  
 
  arma::mat E = arma::zeros<arma::mat>(2 * q + 1, 2 * q + 1);
  E.submat(0, 0, q - 1, q - 1) = diagmat(q_scad(abs(beta0), Lambda) / (abs(beta0) + eps));
  
  
  arma::mat B = inv(H + n * E);
  
 
  arma::mat cov = B * M * B;
  
  return cov;
}

// [[Rcpp::export]]
arma::mat variance_se(arma::vec param, int q, double lambda, double delta, arma::vec u_vec, arma::vec u_prob, arma::mat dataset) {
  int m = dataset.n_rows;
  arma::vec beta = param.subvec(0, q); 
  arma::vec kappa = param.subvec(q + 1, 2 * q);
  
  arma::mat A_m = zeros<arma::mat>(2 * q + 1, 2 * q + 1);
  arma::mat B_m = zeros<arma::mat>(2 * q + 1, 2 * q + 1);
  
  for (int i = 0; i < m; i++) {
    arma::vec X_i = trans(dataset.row(i).subvec(0, q - 1));
    double Z = dataset(i, q);
    int Y = dataset(i, q + 1);
    
    arma::vec phi_i = efficient_score(beta, kappa, lambda, delta, Y, Z, X_i, u_vec, u_prob);
    arma::mat B_i = phi_i * phi_i.t();
    
    Function efficient_score2("efficient_score2");
    arma::mat A_i = -computeJacobian2(efficient_score2, param, lambda, delta, Y, Z, X_i, u_vec, u_prob);
    
    A_m += A_i;
    B_m += B_i;
  }
  
  A_m /= m;
  B_m /= m;
  
  
  arma::mat var_cov_mat = (inv(A_m) * B_m * inv(A_m.t())) / m;
  
  return var_cov_mat;
}

// [[Rcpp::export]]
List CV(arma::vec param_start, int q, double lambda, double delta, arma::vec u_vec,
        arma::vec u_prob, arma::mat dataset, double eps, double tol, int maxiter, arma::vec Lambda_vec, int fold) {
  
  int n_rows = dataset.n_rows;
  int n_Lambda_vec = Lambda_vec.n_elem;
  double Lam_min = -1;
  double cv_min = std::numeric_limits<double>::infinity();
  arma::vec cv_vect;
  
  for (int j = 0; j < n_Lambda_vec; j++) {
    double Lam_temp = Lambda_vec(j);
    double cv_value = 0;
    arma::uvec index_test;
    
    for (int k = 0; k < fold; k++) {
      int n_k = std::floor(n_rows / fold);
      index_test = arma::regspace<arma::uvec>((k * n_k), (k + 1) * n_k - 1);
      //index_test = arma::linspace(k * n_k, (k + 1) * n_k - 1, 1);
      
      if (k == (fold - 1)) {
        //index_test = arma::linspace(k * nk, n - 1, 1);
        index_test = arma::regspace<arma::uvec>((k * n_k), n_rows - 1);
      }
      
      arma::mat dataset_train = dataset;
      dataset_train.shed_rows(index_test);
      arma::mat dataset_test = dataset.rows(index_test);
      
      Function estimating_eq("estimating_eq");
      arma::vec param_cv = Newton_Raphson(estimating_eq, param_start, q, lambda, delta, u_vec, u_prob, dataset_train, eps, tol, maxiter, Lam_temp)(0);
      param_start = param_cv;
      
      arma::vec eeq = as<arma::vec>(estimating_eq(param_cv, q, lambda, delta, u_vec, u_prob, dataset_test));
      cv_value += sum(eeq % eeq);
    }
    
    arma::vec cv_value_vec = arma::vec(1).fill(cv_value);
    cv_vect = arma::join_cols(cv_vect, cv_value_vec);
    
    if (cv_value < cv_min) {
      Lam_min = Lam_temp;
      cv_min = cv_value;
    }
  }
  
  return List::create(Named("Lam_min") = Lam_min,
                      Named("cv_vect") = cv_vect);
}
