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

  MFMMixing() = default;
  ~MFMMixing()= default;

public:
    int get_K() {return state.K;}
    int get_K_plus() {return K_plus;}
    double get_log_alpha() {return state.log_alpha;}
    Eigen::VectorXd get_etas() {return etas; }
    Eigen::VectorXd get_log_etas();

    void set_K_plus(int kk) {K_plus = kk; }
    void set_K(int k_) {state.K = k_; }
    void set_etas(Eigen::VectorXd etas_) {etas = etas_;}

    void update_state(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
                    const std::vector<unsigned int> &allocations) override;

    void update_K(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

    void update_eta(const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values);

    void update_log_alpha();

    int factorial(int n) const;

 protected:
    int K_plus;
    Eigen::VectorXd etas;


};

// Mancano getter e setter
#endif  // BAYESMIX_MFMMIXING_H
