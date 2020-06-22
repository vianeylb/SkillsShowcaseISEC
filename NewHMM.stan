data {
  int<lower = 0> N;  // number of states
  int<lower = 1> T;  // number of observations
  real y[T];
}

parameters {
  simplex[N] theta[N];  // N x N tp
  ordered[N] mu;  // state-dependent parameters
  simplex[N] init;
}

transformed parameters {
  matrix[N, T] log_omega;
  matrix[N, N] Gamma;

  // build log_omega
  for (t in 1:T)
    for (n in 1:N) log_omega[n, t] = normal_lpdf(y[t] | mu[n], 2);

  // build Gamma
  for (n in 1:N) Gamma[n, ] = theta[n]';
}

model {
  mu ~ student_t(3, 0, 1);
  
  target += hmm_marginal(log_omega, Gamma, init);
}