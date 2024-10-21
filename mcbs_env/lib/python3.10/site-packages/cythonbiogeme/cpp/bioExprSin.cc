//-*-c++-*------------------------------------------------------------
//
// File name : bioExprSin.cc
// @date   Thu Sep 21 09:57:05 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioExprSin.h"
#include "bioDebug.h"
#include "bioExceptions.h"
#include <cmath>
#include <sstream>

bioExprSin::bioExprSin(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}

bioExprSin::~bioExprSin() {
  
}

const bioDerivatives* bioExprSin::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							 bioBoolean gradient,
							 bioBoolean hessian) {
  

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;

  const bioDerivatives* child_result = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioReal sin_f = sin(child_result->f) ;
  theDerivatives.f = sin_f ;
  if (gradient) {
    bioReal cos_f = cos(child_result->f) ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives.g[i] = child_result->g[i] * cos_f ;
      if (hessian) {
	for (bioUInt j = 0 ; j <= i ; ++j) {
	  theDerivatives.h[i][j] = -sin_f * child_result->g[i] * child_result->g[j] + cos_f * child_result->h[i][j] ;
	  if (i != j) {
	    theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
	  }
	}
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprSin::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "sin(" << child->print(hp) << ")";
  return str.str() ;
}

