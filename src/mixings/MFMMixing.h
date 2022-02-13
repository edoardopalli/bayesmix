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
  int K; double log_alpha; Eigen::VectorXd etas; //occho a gestire alpha/log_alpha nel modo giusto
};
}

// Mancano i parametri di input delle funzioni
class MFMMixing
    : public BaseMixing<MFMMixing, mfm::State, bayesmix::MFMPrior> {
 public:
  MFMMixing() = default;
  ~MFMMixing()= default;

  //NOTA IMPORTANTE: per come è fatta la nostra classe mixing, lo stato è diverso dai paramentri che passiamo come protomessage
  //quindi le funzioni seguenti vengono overridden solo perchè altrimenti sarebbero pure virtual -> la classe mixing sarebbe così astratta e inutilizzabile
  bool is_conditional() const override { return true; }
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override;


  unsigned int get_num_components() const override {return state.K;}
//dio merda
  //double get_log_alpha() {return state.log_alpha;}
  //Eigen::VectorXd get_etas() {return state.etas; }
  //Eigen::VectorXd get_log_etas();
  //void set_etas(Eigen::VectorXd etas_) {state.etas = etas_;}
  //void set_log_alpha(int log_alpha_) {state.log_alpha = log_alpha_;}

  Eigen::VectorXd mixing_weights(const bool log_,
                                         const bool propto) const override;

  bayesmix::MixingId get_id() const override {
    return bayesmix::MixingId::MFM;
  }


  void set_num_components(const unsigned int k_) override {state.K = k_; }

  int get_K_plus() const override {return K_plus;}

  void set_K_plus(const int kk) override {K_plus = kk; }


  void update_state(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
                    const std::vector<unsigned int> &allocations) override;

  void update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

  void update_eta(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

  //void update_log_alpha();

  float factorial(int n) const;

  void initialize_state() override;
  int sample_BNB();

 protected:
    int K_plus;

};

// Mancano getter e setter
#endif  // BAYESMIX_MFMMIXING_H
