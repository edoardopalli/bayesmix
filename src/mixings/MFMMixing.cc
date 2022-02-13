#include "MFMMixing.h"
#include "src/utils/distributions.h"
#include "src/utils/rng.h"

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include <unsupported/Eigen/MatrixFunctions>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"
#include "src/utils/rng.h"
#include <iostream>
#include <random>

float MFMMixing::factorial(int n) const {
    if ((n == 0) || (n == 1))
        return 1;
    else
        return n * factorial(n - 1);
}

void MFMMixing::update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values) {

  std::cout << "siamo in update_K" << std::endl;
  int kk = get_K_plus();
  //std::cout << kk << std::endl;

  double alpha = exp(state.log_alpha);
  Eigen::VectorXd logprobas(20 - kk + 1);
 // std::cout << logprobas.size() << std::endl;

  Eigen::VectorXd log_prior(20 - kk  +1 );
  float val = (float) 1/(20 - kk + 1);
  //std::cout << val << std::endl;
  for(int i = 0; i < 20 - kk + 1 ; i++){
    log_prior(i) = log(val);
    //std::cout << 1/(20-kk+1) << std::endl;
    //std::cout << log_prior(i) << std::endl;
  }
  //prova con prior uniforme su K

 // std::cout << "Alpha = " << alpha << std::endl;

  auto priorcast = cast_prior();

  for(int j = 0; j < logprobas.size(); j++){

    //std::cout << "Iterazione " << j << std::endl;

    logprobas(j) = (kk)*log(alpha) + log(std::tgamma(1 + j + kk));
    logprobas(j) -= (kk*log(j+kk) + log(std::tgamma(j + 1)));
    //std::cout << logprobas(j) << std::endl;

    double log_new = 0.0;
    for(int i = 0; i < kk; i++){

      //std::cout << unique_values[i] -> get_card() << std::endl;

      long double temp = unique_values[i]->get_card() + (float)alpha/(j+kk);

      long double temp2 = std::tgamma(temp);
      //std::cout << temp2 << std::endl;
      log_new += log(temp2);
      log_new -= log((float)std::tgamma(1 + (float) alpha/(j + kk)));

      //std::cout << log_new << std::endl;
    }

    //std::cout << log_new << std::endl;

    logprobas(j) += log_new;

    //std::cout << logprobas(j) << std::endl;
  }

  if (priorcast->has_fixed_value()) {
    std::cout << "Siamo in fixed_values" << std::endl;
    logprobas += log_prior;
    //std::cout << logprobas << std::endl;
  }
  else if(priorcast -> has_bnb_prior()) {
    std::cout << "Abbiamo la prior su K" << std::endl;
    double rate = priorcast->bnb_prior().k_prior().rate();
    std::cout << rate << std::endl;
    double shape_a = priorcast->bnb_prior().k_prior().shape_a();
    double shape_b = priorcast->bnb_prior().k_prior().shape_b();
    std::cout << shape_b << std::endl;
    //std::cout << "Abbiamo preso le prior" << std::endl;

    //std::cout << "kk"<<kk << std::endl;
    //std::cout << "get_k_plus"<< get_K_plus() << std::endl;
    Eigen::VectorXd priors = bayesmix::evaluate_BNB(rate, shape_a, shape_b, 20);
    //std::cout << priors << std::endl;

    for(int i=0; i < 20 - kk +1; i++){
    std::cout << i << std::endl;
      log_prior(i) = std::log(priors(i+kk-1));
    }
    std::cout << "SIamo fuori dal for" << std::endl;
    logprobas += log_prior;
  }

  //Devo passare la prior di K
  //std::cout << "Siamo al categorical rng" << std::endl;
  //Come input K e K+
  auto &rng = bayesmix::Rng::Instance().get();
  state.K = bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, kk);

  //std::cout << state.K << std::endl;
}

//void MFMMixing::update_log_alpha() {
  //Come input K e K+
  //step 3b
  //step di metropolis hastings
//}

void MFMMixing::update_eta(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values){

  std::cout << "Siamo all'update etas" << std::endl;
  int K = get_num_components();
  double alpha = exp(state.log_alpha);
  int kk = get_K_plus();

  std::cout << "K: " << K << ", k+: " << kk << std::endl;

  Eigen::VectorXd param(K);
  for(int j = 0; j < param.size(); j++){
    param(j) = 0;
  }

  for(int j = 0; j < kk; j++){
      param(j) = unique_values[j]->get_card();
  }

  //std::cout << param << std::endl;

  for(int i = 0; i < K; i++)
    param(i) += (float)(alpha/K);

  //std::cout << param << std::endl;


  auto &rng = bayesmix::Rng::Instance().get();
  state.etas = stan::math::dirichlet_rng(param,rng);

}

void MFMMixing::update_state(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
                             const std::vector<unsigned int> &allocations) {
  std::cout << "Siamo in update state" << std::endl;

  update_K(unique_values);
  //update_log_alpha();
  update_eta(unique_values);

}

/* Eigen::VectorXd MFMMixing::get_log_etas() {
  unsigned int K = get_num_components();
  Eigen::VectorXd log_etas(K);
  for(int i=0; i<log_etas.size(); i++){
    log_etas[i] = log(state.etas[i]);
  }

  return log_etas;
}
*/


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
  //std::cout << state.K << std::endl;
  set_K_plus(state.K);
  state.log_alpha = 0;

  Eigen::VectorXd eta_(state.K);
  float val = (float) 1/state.K;
  for(int i = 0 ; i < eta_.size(); i++){
    eta_(i) = val;
  };
  state.etas = eta_;

  std::cout << state.etas << std::endl;
}

std::shared_ptr<bayesmix::MixingState> MFMMixing::get_state_proto()
    const {
  bayesmix::MFMState state_;
  bayesmix::to_proto(state.etas, state_.mutable_weights());
  state_.set_k(state.K);
  //state_.set_alpha(state.alpha);

  auto out = std::make_shared<bayesmix::MixingState>();
  out->mutable_mfm_state()->CopyFrom(state_);
  return out;
}

void MFMMixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.etas = bayesmix::to_eigen(statecast.mfm_state().weights());
  state.K = statecast.mfm_state().k();
}

Eigen::VectorXd MFMMixing::mixing_weights(const bool log_, const bool propto) const {
  if(log_){
    Eigen::VectorXd log_etas(state.etas.size());
    for(int i =0; i < log_etas.size(); i++){
      log_etas[i] = log(state.etas[i]);
    }
    return log_etas;
  }
  else {
    return state.etas;
  }

}
