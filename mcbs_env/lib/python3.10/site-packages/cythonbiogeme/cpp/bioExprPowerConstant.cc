//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPowerConstant.cc
// Date:   Wed Jul 3 10:20:56 2024
// Author: Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioExprPowerConstant.h"
#include "bioExceptions.h"
#include <cmath>
#include <sstream>
#include "bioDebug.h"
#include "bioConstants.h"

bioExprPowerConstant::bioExprPowerConstant(bioExpression* l, bioReal an_exponent) :
  child(l), exponent(an_exponent) {
  listOfChildren.push_back(l) ;
}

bioExprPowerConstant::~bioExprPowerConstant() {

}

const bioDerivatives* bioExprPowerConstant::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						     bioBoolean gradient,
						     bioBoolean hessian) {

  static const bioReal almost_zero = constants::get_almost_zero();

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;

  bool is_exponent_integer = (std::trunc(exponent) == exponent);

  const bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioReal cf = childResult-> f ;
  if ((!is_exponent_integer) && (cf >= -almost_zero) && (cf < 0.0)) {
      cf = 0.0 ;
  }
  if (exponent == 0) {
    theDerivatives.setDerivativesToZero() ;
    theDerivatives.f = 1.0 ;
    return &theDerivatives ;
  }
  if (exponent == 1.0) {
    theDerivatives.f = cf ;
    if (gradient) {
      for (bioUInt i = 0 ; i < n ; ++i) {
	    theDerivatives.g[i] = childResult->g[i] ;
	    if (hessian) {
          for (bioUInt j = 0 ; j < n ; ++j) {
            theDerivatives.h[i][j] = childResult->h[i][j] ;
          }
        }
      }
	}
    return &theDerivatives ;
  }
  if (exponent == 2.0) {
    theDerivatives.f = childResult->f * childResult->f ;
    if (gradient) {
      for (bioUInt i = 0 ; i < n ; ++i) {
	    if (childResult->g[i] != 0.0) {
	      theDerivatives.g[i] = 2.0 * childResult->f  * childResult->g[i] ;
	    }
	    else {
	      theDerivatives.g[i] = 0.0 ;
	    }
	    if (hessian) {
          for (bioUInt j = i ; j < n ; ++j) {
            theDerivatives.h[i][j] = 0.0 ;
            if (childResult->g[i] != 0.0 && childResult->g[i] != 0.0) {
              theDerivatives.h[i][j] += 2.0 *
                childResult->g[i] *
                childResult->g[j] ;
            }
            if (childResult->h[i][j] != 0.0) {
              theDerivatives.h[i][j] += 2.0 * childResult->h[i][j] ;
            }
            if (i != j) {
              theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
            }
          }
        }
      }
	}
    return &theDerivatives ;
  }

  theDerivatives.f = std::pow(childResult->f, exponent) ;
  if (gradient) {
    bioReal to_k_minus_1 = exponent * std::pow(cf, exponent-1.0);
	for (bioUInt i = 0 ; i < n ; ++i) {
	  if (childResult->g[i] != 0.0) {
	    theDerivatives.g[i] = to_k_minus_1 * childResult->g[i] ;
	  }
	  else {
	    theDerivatives.g[i] = 0.0 ;
	  }
	  if (hessian) {
        for (bioUInt j = i ; j < n ; ++j) {
          theDerivatives.h[i][j] = 0.0 ;
          if (childResult->g[i] != 0.0 && childResult->g[i] != 0.0) {
            theDerivatives.h[i][j] +=
              exponent *
              (exponent - 1.0) *
              std::pow(childResult->f, exponent-2.0) *
              childResult->g[i] *
              childResult->g[j] ;
          }
          if (childResult->h[i][j] != 0.0) {
            theDerivatives.h[i][j] += to_k_minus_1 * childResult->h[i][j] ;
          }
          if (i != j) {
            theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
          }
        }
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprPowerConstant::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "^(" << child->print(hp) << "*" << exponent << ")" ;

  }
  else {
    str << "(" << child->print(hp) << "^" << exponent << ")" ;
  } 
  return str.str() ;
}
