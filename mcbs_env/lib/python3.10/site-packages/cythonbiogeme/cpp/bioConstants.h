#ifndef BIOCONSTANTS_H
#define BIOCONSTANTS_H

#include <limits>
#include <cmath>
#include <iomanip>
#include "bioTypes.h"
#include "bioExceptions.h"

namespace constants {

  // Function to calculate and cache the upper bound on floating values
  //inline bioReal get_upper_bound() {
  //  static const bioReal sqrt_max_float = std::sqrt(std::numeric_limits<bioReal>::max());
  //  return sqrt_max_float;
  //}

  //inline bioReal get_log_upper_bound() {
  //  static const bioReal log_upper = std::log(get_upper_bound());
  //  return log_upper;
  //}

  //inline bioReal get_machine_epsilon() {
  //  return std::sqrt(std::numeric_limits<bioReal>::epsilon());
  //}

  inline bioReal get_almost_zero() {
    static const bioReal sqrt_machine_epsilon = std::sqrt(std::numeric_limits<bioReal>::epsilon());
    return sqrt_machine_epsilon;
  }

}


#endif // BIOCONSTANTS_H