#ifndef BAYESMIX_MIXINGS_BASE_MIXING_HPP_
#define BAYESMIX_MIXINGS_BASE_MIXING_HPP_

#include <memory>

#include "../../proto/cpp/mixings.pb.h"
#include "../hierarchies/BaseHierarchy.hpp"

//! Abstract base class for a generic mixture model

//! This class represents a mixture model object to be used in a BNP iterative
//! algorithm. By definition, a mixture is a probability distribution that
//! integrates over a density kernel to generate the actual distribution for
//! the data. However, in the context of this library, where a clustering
//! structure is generated on the data, a certain mixture translates to a
//! certain way of weighing the insertion of data in old clusters vs the
//! creation of new clusters. Therefore any mixture object inheriting from the
//! class must have methods that provide the probabilities for the two
//! aforementioned events. The class will then have its own parameters, and
//! maybe even prior distributions on them.

class BaseMixing {
 public:
  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~BaseMixing() = default;
  BaseMixing() = default;

  // PROBABILITIES FUNCTIONS
  //! Mass probability for choosing an already existing cluster

  //! \param card Cardinality of the cluster
  //! \param n    Total number of data points
  //! \return     Probability value
  virtual double mass_existing_cluster(const unsigned int card,
                                       const unsigned int n) const = 0;

  //! Mass probability for choosing a newly created cluster

  //! \param n_clust Number of clusters
  //! \param n       Total number of data points
  //! \return        Probability value
  virtual double mass_new_cluster(const unsigned int n_clust,
                                  const unsigned int n) const = 0;

  virtual void update_hypers(
      const std::vector<std::shared_ptr<BaseHierarchy>> &unique_values,
      unsigned int n) = 0;

  virtual void set_state(const google::protobuf::Message &curr) = 0;
  // TODO is it needed?:
  virtual void write_state_to_proto(google::protobuf::Message *out) const = 0;

  virtual std::string get_id() const = 0;
};

#endif  // BAYESMIX_MIXINGS_BASE_MIXING_HPP_
