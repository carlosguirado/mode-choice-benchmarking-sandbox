//-*-c++-*------------------------------------------------------------
//
// File name : validity_check.cc
// @date   Fri Sep 15 10:58:13 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include <cmath>
#include <limits>
#include "validity_check.h"
#include "bioExceptions.h"

bool validity_check(bioReal value, bool raise_exception) {
    bioReal maxDouble = std::numeric_limits<bioReal>::max();
    bioReal sqrtMaxDouble = std::sqrt(maxDouble);
    if (value < sqrtMaxDouble && value > -sqrtMaxDouble) {
      return true ;
    }
    if (raise_exception) {
      throw bioExceptOutOfRange<bioReal>(__FILE__,__LINE__,value,-sqrtMaxDouble,sqrtMaxDouble) ;
    }
    return false ;
}
