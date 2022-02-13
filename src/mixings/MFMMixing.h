//
// Created by Chiara Monfrinotti on 21/12/2021.
//
#ifndef BAYESMIX_MFMMIXING_H
#define BAYESMIX_MFMMIXING_H

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include <math.h>
#include <cmath>
#include "base_mixing.h"
#include "src/utils/proto_utils.h"

#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"


#include "src/hierarchies/abstract_hierarchy.h"

namespace mfm  {
struct State {
  int K; double log_alpha; Eigen::VectorXd etas; //The prior on alpha is for future implementation of a MH step to update it
};
}

class MFMMixing
    : public BaseMixing<MFMMixing, mfm::State, bayesmix::MFMPrior> {
 public:
  MFMMixing() = default;
  ~MFMMixing()= default;

  bool is_conditional() const override { return true; }

  ///Functions to read/write the state of the object on a proto file
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override;


  unsigned int get_num_components() const override {return state.K;}

    ///mixing_weights() -> overrides a virtual member of AbstractMixing: it returns the mixing weights
    Eigen::VectorXd mixing_weights(const bool log_,
                                         const bool propto) const override;

  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::MFM;
  }


  void set_num_components(const unsigned int k_) override {state.K = k_; }

  int get_K_plus() const override {return K_plus;}

  void set_K_plus(const int kk) override {K_plus = kk; }

   ///update_state() -> function called by the algorithm object that performs the update of the state of the mixing,
   ///by running the update of K and of the weights
   void update_state(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
                    const std::vector<unsigned int> &allocations) override;

  ///update_K() -> method that samples from the categorical posterior
  ///the updated value of num_components (elements of the mixture)
  void update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

  ///update_eta() -> sample new values of the vector of etas (i.e. the weights of the mixture)
  ///from a finite dirichlet with proper weights
  void update_eta(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

  //void update_log_alpha(); to be implemented, for MH step

  ///initialize_state() -> overridden function specific for the initialization of the object mixing
  ///It initializes K, etas, K_plus = K
  void initialize_state() override;

  /// Evaluation of a finite dimensional BetaNegativeBinomial PMF
  Eigen::VectorXd evaluate_BNB(int rate, double shape_a, double shape_b, int K_max);


protected:
    int K_plus;

};

// Mancano getter e setter
#endif  // BAYESMIX_MFMMIXING_H
