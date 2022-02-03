#include "MFMMixing.h"
#include "src/utils/distributions.h"
#include "src/utils/rng.h"


int MFMMixing::factorial(int n) const {
    if ((n == 0) || (n == 1))
        return 1;
    else
        return n * factorial(n - 1);
}

void MFMMixing::update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values) {

  int kk = get_K_plus();
  double alpha = exp(get_log_alpha());
  //prova con prior uniforme su K
  Eigen::VectorXd log_prior(100 - kk + 1, 1/(100 - kk +1));
  Eigen::VectorXd logprobas(100 - kk + 1);
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
  auto &rng = bayesmix::Rng::Instance().get();
  state.K = bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);

  //DA FARE: eval prior BNB p(K)
  //         come prendere Nk

}

void MFMMixing::update_log_alpha() {
  //Come input K e K+
  //step 3b
  //step di metropolis hastings
}

void MFMMixing::update_eta(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values){
  int K = get_K();
  int alpha = exp(get_log_alpha());
  Eigen::VectorXd param(K);
  for(int j = 0; j < K; j++){
      param[j] = unique_values[j]->get_card() + alpha/K;
  }
  auto &rng = bayesmix::Rng::Instance().get();
  etas = stan::math::dirichlet_rng(param,rng);
}

void MFMMixing::update_state(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
                             const std::vector<unsigned int> &allocations) {
  update_K(unique_values);
  update_eta(unique_values);
  update_log_alpha();
}

Eigen::VectorXd MFMMixing::get_log_etas() {
  int K = get_K();
  Eigen::VectorXd log_etas(K);
  for(int i=0; i<log_etas.size(); i++){
    log_etas[i] = log(etas[i]);
  }

  return log_etas;
}