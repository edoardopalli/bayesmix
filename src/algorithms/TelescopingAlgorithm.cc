//
// Created by aleda on 21/12/2021.
//

#include "TelescopingAlgorithm.h"


//We don't find HierarchyID_Name
void TelescopingAlgorithm::print_startup_message() const {
    std::string msg = "Running Telescoping algorithm with " +
                      bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                      " hierarchies, " +
                      bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
    std::cout << msg << std::endl;
}

void remove_empty(const unsigned int idx) {
    //Relabel allocations
    for(auto &c : allocations){
        if(c>idx){
            c -= 1;
        }
    }
    //remove cluster
    unique_values.erase(unique_values.begin() + idx)
}

virtual void TelescopingAlgorithm::sample_allocations() override{
    //step 1a of algorithm

    for(int i = 0; i < data.rows(); i++){
        Eigen::VectorXd log_probas =
                mixing -> get_mixing_weights(true, true);
        for(int j = 0; j < log_probas.size(); j++){
            log_probas(j) +=
                    unique_values[j] -> get_like_lpdf(data.row(i))
        }

        // Draw a NEW value for datum allocation
        unsigned int c_new =
            bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);
        unsigned int c_old = allocations[i];
        if (c_new != c_old) {
        allocations[i] = c_new;
        // Remove datum from old cluster, add to new
        // We don't need to update cluster parameters when adding a new datum; update_hierarchy_parameter is expected to be false.
        unique_values[c_old]->remove_datum(
            i, data.row(i), update_hierarchy_params());
        unique_values[c_new]->add_datum(
            i, data.row(i), update_hierarchy_params());
        }
    }
// We have to save K+, the number of non-empty (filled) components
    KK = compute_KK(unique_values);
    mixing->set_K_plus(KK);
}

//Compute k_plus using get_card method of hierarchies (unique values); we then use function remove_empty inj order to remove empty clusters and perform relabeling
unsigned int TelescopingAlgorithm::compute_KK (std::vector<std::shared_ptr<AbstractHierarchy>> unique_values) {
    for(int j=0; j < unique_values.size(); j++){
        int count = 0;
        if(unique_values[j]->get_card() > 0){
            count += 1;
        }
        else {
            remove_empty(j);
            j -= 1;
        }
    }
}

virtual void sample_unique_values() {
    for (auto &un : unique_values) {
        un->sample_full_cond(!update_hierarchy_params());
    }
}

