#include <math.h>

#include <fstream>
#include <iostream>

#include "src/includes.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Running run.cpp" << std::endl;

  // Console parameters (temporarily assigned at compile-time)
  std::string type_mixing = "PY";
  std::string type_hier = "NNW";
  std::string type_algo = "N2";
  std::string datafile = "resources/data_multi.csv";
  std::string gridfile = "resources/grid_multi.csv";
  unsigned int init = 0;

  // Create factories and objects
  Factory<BaseMixing> &factory_mixing = Factory<BaseMixing>::Instance();
  Factory<HierarchyBase> &factory_hier = Factory<HierarchyBase>::Instance();
  Factory<Algorithm> &factory_algo = Factory<Algorithm>::Instance();
  auto mixing = factory_mixing.create_object(type_mixing);
  auto hier = factory_hier.create_object(type_hier);
  auto algo = factory_algo.create_object(type_algo);

  // Other objects
  Eigen::MatrixXd data = bayesmix::read_eigen_matrix(datafile);
  Eigen::MatrixXd grid = bayesmix::read_eigen_matrix(gridfile);

  // STUFF:
  // Object allocation
  algo->set_mixing(mixing);
  algo->set_data_and_initial_clusters(data, hier, init);

  // Execution
  algo->print_id();
  algo->get_mixing_id();
  algo->get_hier_id();

  std::cout << "End of run.cpp" << std::endl;
  return 0;
}