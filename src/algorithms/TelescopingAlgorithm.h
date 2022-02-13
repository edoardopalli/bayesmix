//
// Created by aleda on 21/12/2021.
//

#ifndef UNTITLED1_TELESCOPINGALGORITHM_H
#define UNTITLED1_TELESCOPINGALGORITHM_H

#include <google/protobuf/message.h>
#include <lib/progressbar/progressbar.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "base_algorithm.h"
#include "conditional_algorithm.h"
#include "algorithm_id.pb.h"
#include "algorithm_params.pb.h"
#include "algorithm_state.pb.h"
#include "src/collectors/base_collector.h"
#include "src/hierarchies/base_hierarchy.h"
#include "src/mixings/MFMMixing.h"


/* This child class, inheriting from BaseAlgorithm has the goal to implement a new MCMC Algorithm,
 * Telescoping Algorithm, in order to make posterior inference on the parameters in a Mixture o Finite
 * Mixtures Model (MFM, see class MFMMixing).
 * The Algorithm and the necessary theory is presented in FRÜHWIRTH-SCHNATTER, S., MALSINER-WALLI, G. AND GRÜN, B.
 * 2020, Generalized Mixture of Finite Mixtures and Telescoping Sampling.
 */
class TelescopingAlgorithm : public ConditionalAlgorithm{
public:
    TelescopingAlgorithm() = default;
    ~TelescopingAlgorithm() = default;

   public:
    bayesmix::AlgorithmId get_id() const  override{
      return bayesmix::AlgorithmId::Telescoping;
    }
    //! Prints a message at the beginning of `run()`
    void print_startup_message() const override;

    //! Performs Gibbs sampling sub-step for all allocation values
    void sample_allocations() override;

    //! Performs Gibbs sampling sub-step for all unique values
    void sample_unique_values() override;

    //! Counts the filled components (hierarchies)
    unsigned int compute_KK(std::vector<std::shared_ptr<AbstractHierarchy>> &);

    //!Adds empty hierarchies (if any) after simulating K >= K+
    void add_new_clust();

    //!Runs the whole algorithm
    void step() override;

    //!Necessary to force this class to be non abstract (otherwise it would inherit a virtual = 0 method)
    void sample_weights() override {}


};


#endif //UNTITLED1_TELESCOPINGALGORITHM_H
