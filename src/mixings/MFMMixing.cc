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

///Note: The value k_max has been considered to be 20.

void MFMMixing::update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values) {

  int kk = get_K_plus();

  double alpha = exp(state.log_alpha);
  Eigen::VectorXd logprobas(20 - kk + 1);

  Eigen::VectorXd log_prior(20 - kk  +1 );
  float val = (float) 1/(20 - kk + 1);
  for(int i = 0; i < 20 - kk + 1 ; i++){
    log_prior(i) = log(val);
    }

  auto priorcast = cast_prior();

  for(int j = 0; j < logprobas.size(); j++){

    logprobas(j) = (kk)*log(alpha) + log(std::tgamma(1 + j + kk));
    logprobas(j) -= (kk*log(j+kk) + log(std::tgamma(j + 1)));

    double log_new = 0.0;
    for(int i = 0; i < kk; i++){

      long double temp = unique_values[i]->get_card() + (float)alpha/(j+kk);
      long double temp2 = std::tgamma(temp);
      log_new += log(temp2);
      log_new -= log((float)std::tgamma(1 + (float) alpha/(j + kk)));

    }
    logprobas(j) += log_new;
  }

  if (priorcast->has_fixed_value()) {
    logprobas += log_prior;
  }
  else if(priorcast -> has_bnb_prior()) {
    double rate = priorcast->bnb_prior().k_prior().rate();
    double shape_a = priorcast->bnb_prior().k_prior().shape_a();
    double shape_b = priorcast->bnb_prior().k_prior().shape_b();

    Eigen::VectorXd priors = evaluate_BNB(rate, shape_a, shape_b, 20);

    for(int i=0; i < 20 - kk +1; i++){
      log_prior(i) = std::log(priors(i+kk-1));
    }
    logprobas += log_prior;
  }

  auto &rng = bayesmix::Rng::Instance().get();
  state.K = bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, kk);

}

void MFMMixing::update_eta(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values){

  int K = get_num_components();
  double alpha = exp(state.log_alpha);
  int kk = get_K_plus();

  Eigen::VectorXd param(K);
  for(int j = 0; j < param.size(); j++){
    param(j) = 0;
  }

  for(int j = 0; j < kk; j++){
      param(j) = unique_values[j]->get_card();
  }

  for(int i = 0; i < K; i++)
    param(i) += (float)(alpha/K);

  auto &rng = bayesmix::Rng::Instance().get();
  state.etas = stan::math::dirichlet_rng(param,rng);

}

void MFMMixing::update_state(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
                             const std::vector<unsigned int> &allocations) {

  update_K(unique_values);
  //update_log_alpha();
  update_eta(unique_values);

}

void MFMMixing::initialize_state() {
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

Eigen::VectorXd MFMMixing::evaluate_BNB(int rate, double shape_a, double shape_b, int K_max) {
    Eigen::VectorXd probas(K_max);
    for(int i=0; i<K_max; i++){
        probas[i] = (std::beta(rate + i, shape_a + shape_b)/std::beta(rate,shape_a)) * (tgamma(i + shape_b)/(tgamma(i+1)*tgamma(shape_b)));
    }
    return probas;

}