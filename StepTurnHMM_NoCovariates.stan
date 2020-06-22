//code written with Théo Michelot (University of St Andrews)
data {
    int<lower=0> T; // length of the time series
    int ID[T]; // track identifier
    vector[T] steps; // step lengths
    vector[T] angles; // turning angles
    int<lower=1> N; // number of states
}

parameters {
    positive_ordered[N] mu; // mean of gamma - ordered
    vector<lower=0>[N] sigma; // SD of gamma
    // unconstrained angle parameters
    vector[N] xangle;
    vector[N] yangle;
    //tpm
    simplex[N] gamma[N];
}  

transformed parameters {
    vector<lower=0>[N] shape;
    vector<lower=0>[N] rate;
    vector<lower=-pi(),upper=pi()>[N] loc;
    vector<lower=0>[N] kappa;

    
    // derive turning angle mean and concentration
    for(n in 1:N) {
        loc[n] = atan2(yangle[n], xangle[n]);
        kappa[n] = sqrt(xangle[n]*xangle[n] + yangle[n]*yangle[n]);
    }
    
    // transform mean and SD to shape and rate
    for(n in 1:N)
        shape[n] = mu[n]*mu[n]/(sigma[n]*sigma[n]);
    
    for(n in 1:N)
        rate[n] = mu[n]/(sigma[n]*sigma[n]);
}

model {
    vector[N] logp;
    vector[N] logptemp;
    matrix[N,N] log_gamma_tr;
    
    // priors
    mu[1] ~ normal(1, 1);
    mu[2] ~ normal(5, 1);
    sigma ~ student_t(3, 0, 1);
    xangle[1] ~ normal(-0.5, 1); // equiv to concentration when yangle = 0
    xangle[2] ~ normal(2, 2);
    yangle ~ normal(0, 0.5); // zero if mean angle is 0 or pi
      
    // transpose
    for(i in 1:N)
        for(j in 1:N)
            log_gamma_tr[j,i] = log(gamma[i,j]);

    // likelihood computation
    for (t in 1:T) {
        // initialise forward variable if first obs of track
        if(t==1 || ID[t]!=ID[t-1])
            logp = rep_vector(-log(N), N);
        
        for (n in 1:N) {
            logptemp[n] = log_sum_exp(to_vector(log_gamma_tr[n]) + logp);
            
            //to account for missing values
            if(steps[t]>=0)
                logptemp[n] = logptemp[n] + gamma_lpdf(steps[t] | shape[n], rate[n]);           
            //to account for missing values    
            if(angles[t]>=(-pi()))
                logptemp[n] = logptemp[n] + von_mises_lpdf(angles[t] | loc[n], kappa[n]);    
        }
        logp = logptemp;
        
        // add log forward variable to target at the end of each track
        if(t==T || ID[t+1]!=ID[t])
            target += log_sum_exp(logp);
    }
}



