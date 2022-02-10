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
#include "abstract_mixing.h"
#include "base_mixing.h"

#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"


#include "src/hierarchies/abstract_hierarchy.h"

namespace mfm  {
struct state {
  int K; double log_alpha;
};
}

// Mancano i parametri di input delle funzioni
class MFMMixing
    : public BaseMixing<MFMMixing, mfm::state, bayesmix::MFMPrior> {
 public:
  MFMMixing() = default;
  ~MFMMixing()= default;

  //NOTA IMPORTANTE: per come è fatta la nostra classe mixing, lo stato è diverso dai paramentri che passiamo come protomessage
  //quindi le funzioni seguenti vengono overridden solo perchè altrimenti sarebbero pure virtual -> la classe mixing sarebbe così astratta e inutilizzabile
  bool is_conditional() const override { return true; }
  void set_state_from_proto(const google::protobuf::Message &state_) override {}
  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override {}


  int get_K() {return state.K;}
  int get_K_plus() {return K_plus;}
  double get_log_alpha() {return state.log_alpha;}
  Eigen::VectorXd get_etas() {return etas; }
  Eigen::VectorXd get_log_etas();
  bayesmix::MixingId get_id() const override { return bayesmix::MixingId::MFM; }

  void set_K_plus(int kk) {K_plus = kk; }
  void set_K(int k_) {state.K = k_; }
  void set_etas(Eigen::VectorXd etas_) {etas = etas_;}
  void set_log_alpha(int log_alpha_) {state.log_alpha = log_alpha_;}
  void update_state(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
                  const std::vector<unsigned int> &allocations) override;

  void update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

  void update_eta(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

  void update_log_alpha();

  int factorial(int n) const;

  void initialize_state() override;
  int sample_BNB();

 protected:
    int K_plus;
    Eigen::VectorXd etas;
};

// Mancano getter e setter
#endif  // BAYESMIX_MFMMIXING_H
