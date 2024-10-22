//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLog.cc
// @date   Tue Apr 17 12:16:32 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprLog.h"
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioConstants.h"
#include <cmath>
#include <sstream>

bioExprLog::bioExprLog(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}

bioExprLog::~bioExprLog() {
  
}

const bioDerivatives* bioExprLog::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							 bioBoolean gradient,
							 bioBoolean hessian) {
  

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;

  static const bioReal almost_zero = constants::get_almost_zero();


  const bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioReal cf = childResult->f ;
  if ((cf >= -almost_zero) && (cf < 0)) {
    cf = 0.0 ;
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

bioString bioExprLog::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "log(" << child->print(hp) << ")";
  return str.str() ;
}

