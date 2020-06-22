#p(s_T | y_1:T)
#p(s_t|s_t+1, y_1:T) \propto α(s_t)p(s_t+1|s_t)

log_sum_exp <- function(x1, x2){
  
  msf = max(x1 + x2)
  exps = sum(exp(x1+x2 - msf))
  
  lse = msf + log(exps)
  
  return(lse)
  
}

ffbs_stepturn <- function(N, TT, log_gamma_tr, gamma, log_delta, steps, shape, rate, angles, kappa, loc){
  
  stateDraws <- rep(NA, TT)
  logalpha <- matrix(NA, nrow=TT, ncol=N)
  
  ## forward variables
  for(n in 1:N){
    logalpha[1,n] = log_delta[n] 
    if(steps[1]>0)
      logalpha[1,n] = logalpha[1,n] + log(dgamma(x=steps[1], shape = shape[n], rate=rate[n])) 
    if(angles[1]>= -pi){
      logalpha[1,n] = logalpha[1,n] + log(dvm(theta=angles[1], mu = loc[n], kappa = kappa[n]))
    }
    
  }
  
  for(t in 2:TT){
    for(n in 1:N){
      logalpha[t,n] = log_sum_exp(log_gamma_tr[n,,t], logalpha[t-1,]) 
      if(steps[t]>0)
        logalpha[t,n] = logalpha[t,n] + log(dgamma(x=steps[t], shape = shape[n], rate=rate[n])) 
      if(angles[t]>= -pi)
        logalpha[t,n] = logalpha[t,n] + log(dvm(theta=angles[t], mu = loc[n], kappa = kappa[n]))
    }    
  }
  
  ## log likelihood
  llk = log_sum_exp(logalpha[TT,], rep(0, N))  
  
  ## state draws from the joint posterior distribution
  stateDraws[TT] = sample(x = 1:N, size=1, prob = exp(logalpha[TT,] - llk))
  
  for(t0 in (TT-1):1){
    
    t0prob_unnorm <- logalpha[t0,]  + log(gamma[,stateDraws[t0+1],t0])
    stateDraws[t0] = sample(x=1:N, size=1, 
                            prob = exp(t0prob_unnorm  - log_sum_exp(t0prob_unnorm, rep(0, N))))
    #print(exp(t0prob_unnorm  - log_sum_exp(t0prob_unnorm, rep(0, N))))
    
  }
  
  
  return(stateDraws)
  
}



