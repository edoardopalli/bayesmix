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

namespace mfm  {
struct state {
  int K; int K_plus; VectorXd etas; double alpha;
};
}

// Mancano i parametri di input delle funzioni
class MFMMixing
    : public BaseMixing<MFMMixing, mfm::state, bayesmix::BNBprior> {

  int get_K() {return K;}
  int get_K_plus() {return K_plus;}

  void set_K_plus(int kk) {K_plus = kk; }
  void set_K(int k_) {K = k_; }

  void update_state();

  void update_K();

  void update_eta();

  void update_alpha();

};

// Mancano getter e setter
#endif  // BAYESMIX_MFMMIXING_H
