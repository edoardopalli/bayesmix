//
// Created by aleda on 21/12/2021.
//

#ifndef UNTITLED1_TELESCOPINGALGORITHM_H
#define UNTITLED1_TELESCOPINGALGORITHM_H

#include <lib/progressbar/progressbar.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "base_algorithm.h"

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
class TelescopingAlgorithm : public BaseAlgorithm {
public:
    TelescopingAlgorithm() = default;
    ~TelescopingAlgorithm() = default;

    // Implementing the same as in conditional algorithm
    virtual bool is_conditional() const override { return true; }

    virtual bool requires_conjugate_hierarchy() const { return false; }

    bayesmix::AlgorithmId get_id() const override {
        return bayesmix::AlgorithmId::Telescoping;
    }

    //! Prints a message at the beginning of `run()`
    virtual void print_startup_message() const override;

    //! Performs Gibbs sampling sub-step for all allocation values
    virtual void sample_allocations() override;

    //! Performs Gibbs sampling sub-step for all unique values
    virtual void sample_unique_values() override;

    unsigned int compute_KK(std::vector<std::shared_ptr<AbstractHierarchy>>);

    void remove_empty(const unsigned int);

    void add_new_clust();

    void step() override;

};


#endif //UNTITLED1_TELESCOPINGALGORITHM_H
