//-*-c++-*------------------------------------------------------------
//
// File name : bioExprIntegrate.cc
// @date   Wed May  9 17:38:29 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprIntegrate.h"
#include <sstream>
#include <cmath>
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioExprGaussHermite.h"
#include "bioConstants.h"



bioExprIntegrate::bioExprIntegrate(bioExpression* c, bioUInt id) :
  child(c), rvId(id) {
  listOfChildren.push_back(c) ;
}
bioExprIntegrate::~bioExprIntegrate() {

}

const bioDerivatives* bioExprIntegrate::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							  bioBoolean gradient,
							  bioBoolean hessian) {

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  theDerivatives.resize(literalIds.size()) ;

  bioExprGaussHermite theGh(child, literalIds, rvId, gradient, hessian) ;
  bioGaussHermite theGhAlgo(&theGh) ;
  std::vector<bioReal> r = theGhAlgo.integrate() ;
  theDerivatives.f = r[0] ;
  bioUInt n = literalIds.size() ;
  if (gradient) {
    for (bioUInt j = 0 ; j < n ; ++j) {
	   theDerivatives.g[j] = r[j+1] ;
    }
  }
  if (hessian) {
    bioUInt index = 1 + theDerivatives.g.size() ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	    theDerivatives.h[i][j] = theDerivatives.h[j][i] = r[index] ;
	  }
	}
	++index ;
  }
  return &theDerivatives ;
}

bioString bioExprIntegrate::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "Integrate(" << child->print(hp) << "," << rvId << ")" ;
  return str.str() ;

}
