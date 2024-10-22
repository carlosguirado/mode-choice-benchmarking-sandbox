//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLogzero.cc
// @date   Mon Oct 24 09:48:09 2022
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprLogzero.h"
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioConstants.h"

#include <cmath>
#include <sstream>

bioExprLogzero::bioExprLogzero(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}

bioExprLogzero::~bioExprLogzero() {
  
}

const bioDerivatives* bioExprLogzero::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							     bioBoolean gradient,
							     bioBoolean hessian) {
  
theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;

  static const bioReal almost_zero = constants::get_almost_zero();



  const bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioReal cf = childResult->f ;
  if (cf >= -almost_zero && cf <= 0) {
    cf = 0.0 ;
  }
  if (cf == 0.0) {
    theDerivatives.f = 0.0 ;
    if (gradient) {
      for (bioUInt i = 0 ; i < n ; ++i) {
        theDerivatives.g[i] = 0.0 ;
        if (hessian) {
          theDerivatives.h[i][i] = 0 ;
	      for (bioUInt j = 0 ; j < n ; ++j) {
	        theDerivatives.h[i][j] = theDerivatives.h[j][i] = 0.0 ;
	      }
	    }
	  }
	}
	return &theDerivatives ;
  }

  theDerivatives.f = log(cf) ;
  if (gradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives.g[i] = childResult->g[i] / cf ;
      if (hessian) {
	    for (bioUInt j = i ; j < n ; ++j) {
	      bioReal fsquare = cf * cf ;
	      theDerivatives.h[i][j] = childResult->h[i][j] / cf - childResult->g[i] *  childResult->g[j] / fsquare ;
	      if (i != j) {
              theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
          }
	    }
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprLogzero::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "logzero(" << child->print(hp) << ")";
  return str.str() ;
}

