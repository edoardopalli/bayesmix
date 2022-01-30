//
// Created by Chiara Monfrinotti on 21/12/2021.
//

//guarda bene dririchlet mixing


#ifndef BAYESMIX_MFMMIXING_H
#define BAYESMIX_MFMMIXING_H

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <math.h>
#include "abstract_mixing.h"
#include "base_mixing.h"
#include "dirichlet_mixing.h"

#include "src/hierarchies/abstract_hierarchy.h"

namespace mfm  {
struct state {
  int K; int K_plus; Eigen::VectorXd etas; double alpha;
};
}

// Mancano i parametri di input delle funzioni
class MFMMixing
    : public BaseMixing<MFMMixing, mfm::state, bayesmix::DPPrior> {

  MFMMixing() = default;
  ~MFMMixing()= default;

  int get_K() {return state.K;}
  int get_K_plus() {return state.K_plus;}

  void set_K_plus(int kk) {state.K_plus = kk; }
  void set_K(int k_) {state.K = k_; }

  void update_state(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
                    const std::vector<unsigned int> &allocations) override;

  void update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

  void update_eta(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

  void update_alpha();

};

// Mancano getter e setter
#endif  // BAYESMIX_MFMMIXING_H
