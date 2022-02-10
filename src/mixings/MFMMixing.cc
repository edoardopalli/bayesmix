#include "MFMMixing.h"
#include "src/utils/distributions.h"
#include "src/utils/rng.h"

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"
#include "src/utils/rng.h"
#include <iostream>
#include <random>

int MFMMixing::factorial(int n) const {
    if ((n == 0) || (n == 1))
        return 1;
    else
        return n * factorial(n - 1);
}

void MFMMixing::update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values) {

  int kk = get_K_plus();
  double alpha = exp(get_log_alpha());
  Eigen::VectorXd logprobas(100 - kk + 1);
  Eigen::VectorXd log_prior(100 - kk + 1, 1/(100 - kk + 1));
  //prova con prior uniforme su K

  auto priorcast = cast_prior();

  double rate = priorcast->bnb_prior().k_prior().rate();
  double shape_a = priorcast->bnb_prior().k_prior().shape_a();
  double shape_b = priorcast->bnb_prior().k_prior().shape_b();

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

  if (priorcast->has_fixed_value()) {
    logprobas += log_prior;
  }
  else {
    Eigen::VectorXd priors = bayesmix::evaluate_BNB(rate, shape_a, shape_b, 99);
    for(int i = kk-1; i < 100; i++){
      log_prior[i] = log(priors[i]);
    }
    logprobas += log_prior;
  }

  //Devo passare la prior di K

  //Come input K e K+
  auto &rng = bayesmix::Rng::Instance().get();
  state.K = bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, kk);

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
  update_log_alpha();
  update_eta(unique_values);

}

Eigen::VectorXd MFMMixing::get_log_etas() {
  int K = get_K();
  Eigen::VectorXd log_etas(K);
  for(int i=0; i<log_etas.size(); i++){
    log_etas[i] = log(etas[i]);
  }

  return log_etas;
}



int MFMMixing::sample_BNB() {
  auto priorcast = cast_prior();

  //Genero rng
  auto &rng = bayesmix::Rng::Instance().get();
  //Per sampling da una NB
  std::default_random_engine generator;

  double rate = priorcast->bnb_prior().k_prior().rate();
  double shape_a = priorcast->bnb_prior().k_prior().shape_a();
  double shape_b = priorcast->bnb_prior().k_prior().shape_b();
  if (rate <= 0) {
    throw std::invalid_argument("Rate parameter must be > 0");
  }
  if (shape_a <= 0) {
    throw std::invalid_argument("Shape a parameter must be > 0");
  }
  if (shape_b <= 0) {
    throw std::invalid_argument("Shape b parameter must be > 0");
  }

  //p ~ beta(a,b), p=y1/(y1+y2), y1 ~ gamma(a,d), y2 ~ gamma(b,d)
  //per ogni d, d=1
  //p ~ beta(a,b), X ~ negBin(r,p) -> X ~ BNB(r,a,b)
  //double p = stan::math::beta_rng(rng, shape_a, shape_b);
  std::gamma_distribution<double> gamma1(shape_a,1);
  double y1 = gamma1(generator);
  std::gamma_distribution<double> gamma2(shape_b,1);
  double y2 = gamma2(generator);
  double p = y1/(y1+y2);
  //p è una Beta(rate_a, rate_b)

  std::negative_binomial_distribution<int> BNB(rate,p);
  return BNB(generator);

}

void MFMMixing::initialize_state() {
    auto priorcast = cast_prior();
    if (priorcast->has_fixed_value()) {
      std::default_random_engine generator;
      std::uniform_int_distribution<int> uniform(1, 100);

      //mettiamo kmax uguale a 100, da controllare
      set_K(uniform(generator));
    }
    else {
      set_K(sample_BNB());
    }
    set_log_alpha(0);
}