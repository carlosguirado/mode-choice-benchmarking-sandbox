//-*-c++-*------------------------------------------------------------
//
// File name : bioExprCos.cc
// @date   Thu Sep 21 10:10:24 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioExprCos.h"
#include "bioDebug.h"
#include "bioExceptions.h"
#include <cmath>
#include <sstream>

bioExprCos::bioExprCos(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}

bioExprCos::~bioExprCos() {
  
}

const bioDerivatives* bioExprCos::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							 bioBoolean gradient,
							 bioBoolean hessian) {
  

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;

  const bioDerivatives* child_result = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioReal cos_f = std::cos(child_result->f) ;
  theDerivatives.f = cos_f ;
  if (gradient) {
    bioReal sin_f = std::sin(child_result->f) ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives.g[i] = - child_result->g[i] * sin_f ;
      if (hessian) {
	    for (bioUInt j = 0 ; j <= i ; ++j) {
	      theDerivatives.h[i][j] = -cos_f * child_result->g[i] * child_result->g[j] - sin_f * child_result->h[i][j] ;
	      if (i != j) {
	        theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
	      }
	    }
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprCos::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "cos(" << child->print(hp) << ")";
  return str.str() ;
}

