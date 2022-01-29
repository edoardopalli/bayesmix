#include "MFMMixing.h"


void MFMMixing::update_K() {

  kk = mfm::state.K_plus;
  alpha = mfm::state.alpha;
  //prova con prior uniforme su K
  std::vector<double> log_prior(100 - kk + 1, 1/(100 - kk +1))
  std::vector<double> logprobas(100 - kk + 1);
  //Devo passare la prior di K
  //K-1 -> BNB(1; 4; 3)
  //Vedi in hierarchy_prior.proto
  //message BNBPrior
    //alpha = 1;
    //a_pi = 4;
    //b_pi = 3;

  for(int j = 0; logprobas.size(); j++){
      logprobas[j] = (kk)*log(alpha) + log(factorial(j + kk));
      logprobas[j] -= (kk*log(j+kk) + log(factorial(j)));

      double log_new = 0.0;
      for(int i = 0; i < kk; i++){
          log_new += log(std::tgamma(unique_values[i]->get_card()) + alpha/(j+kk));
          log_new -= log(std::tgamma(1 + alpha/(j + kk)));
      }
      logprobas[j] += log_new;
  }
  logprobas += log_prior;
  //Come input K e K+
  state.K=categorical_rng(logprobas, rng, 0);

  //DA FARE: eval prior BNB p(K)
  //         come prendere Nk

}

void MFMMixing::update_alpha() {
  //Come input K e K+
  //step 3b
  //step di metropolis hastings
}

void MFMMixing::update_eta() {
  //calcolo (e1,..,ek):=params
  state.eta = stan::math::dirichlet_rng(param;rng);
}

void MFMMixing::update_state() {
  update_K();
  update_eta();
  update_alpha();
}
