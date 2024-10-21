//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPower.cc
// @date   Fri Apr 13 12:20:46 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprPower.h"
#include <cmath>
#include <sstream>
#include "bioDebug.h"
#include "bioConstants.h"
#include "bioExceptions.h"

bioExprPower::bioExprPower(bioExpression* l, bioExpression* r) :
  left(l), right(r) {
  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;

}

bioExprPower::~bioExprPower() {

}

const bioDerivatives* bioExprPower::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						     bioBoolean gradient,
						     bioBoolean hessian) {

  static const bioReal almost_zero = constants::get_almost_zero();


  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;

  const bioDerivatives* leftResult = left->getValueAndDerivatives(literalIds,gradient,hessian) ;

  const bioDerivatives* rightResult = right->getValueAndDerivatives(literalIds,gradient,hessian) ;

  bioReal left_value(leftResult->f) ;
  if ((left_value >= -almost_zero) && (left_value < 0)) {
    left_value = 0.0 ;
  }


  theDerivatives.f = pow(left_value,rightResult->f) ;
  if (gradient) {
    std::vector<bioReal> G(n, 0.0) ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives.g[i] = 0.0 ;
      if (theDerivatives.f != 0.0) {
	    if ((leftResult->g[i] != 0.0) && (rightResult->f != 0.0)) {
	      bioReal term = leftResult->g[i] * rightResult->f / left_value ;
	      G[i] += term ;
	    }
	    if (rightResult->g[i] != 0.0) {
	      if (rightResult->g[i] != 0) {
	        G[i] += rightResult->g[i] * log(left_value) ;
	      }
	    }
	    theDerivatives.g[i] = theDerivatives.f * G[i] ;
      }
    }
    if (hessian) {
      for (bioUInt i = 0 ; i < n ; ++i) {
	    for (bioUInt j = i ; j < n ; ++j) {
	      bioReal v = G[i] * theDerivatives.g[j] ;
	      if (theDerivatives.f != 0 && rightResult != NULL) {
	        bioReal term(0.0) ;
	        bioReal hright = rightResult->h[i][j] ;
	        if (hright != 0.0) {
	          term += hright * log(left_value) ;
	        }
	        if (leftResult->g[j] != 0.0 && rightResult->g[i] != 0.0) {
	          term += leftResult->g[j] * rightResult->g[i] / left_value ;
	        }
	        if (leftResult->g[i] != 0.0 && rightResult->g[j] != 0.0) {
	          term += leftResult->g[i] * rightResult->g[j] / left_value ;
	        }
	        if (leftResult->g[i] != 0.0 && leftResult->g[j] != 0.0) {
	          bioReal asquare = leftResult->f * left_value ;
	          term -= leftResult->g[i] * leftResult->g[j] * rightResult->f / asquare ;
	        }
	        bioReal hleft = leftResult->h[i][j] ;
	        if (hleft != 0.0) {
	          term += hleft * rightResult->f / left_value ;
	        }
	        if (term != 0.0) {
	          v += term * theDerivatives.f ;
	        }
	      }
	      if (i == j) {
	        theDerivatives.h[i][i] = v ;
	      }
	      else {
	        theDerivatives.h[i][j] = theDerivatives.h[j][i] = v ;
	      }
	    }
      }
    }
  }
  return &theDerivatives ;
}



bioString bioExprPower::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "^(" << left->print(hp) << "*" << right->print(hp) << ")" ;

  }
  else {
    str << "(" << left->print(hp) << "^" << right->print(hp) << ")" ;
  } 
  return str.str() ;
}
