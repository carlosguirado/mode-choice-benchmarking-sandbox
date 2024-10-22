//-*-c++-*------------------------------------------------------------
//
// File name : bioExprConditionalSum.cc
// @date   Wed Sep 20 19:00:11 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include <sstream>
#include "bioDebug.h"
#include "bioExprConditionalSum.h"
#include "bioExceptions.h"

bioExprConditionalSum::bioExprConditionalSum(std::unordered_map<bioExpression*, bioExpression*> e) :
  expressions(e) {
  for (const auto& pair : expressions) {
    listOfChildren.push_back(pair.first);
    listOfChildren.push_back(pair.second);
  }
}

bioExprConditionalSum::~bioExprConditionalSum() {

}

const bioDerivatives* bioExprConditionalSum::getValueAndDerivatives(std::vector<bioUInt> literalIds,
								    bioBoolean gradient,
								    bioBoolean hessian) {

  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,__LINE__,"If the hessian is needed, the gradient must be computed") ;
  }

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;
  theDerivatives.resize(literalIds.size()) ;
  std::size_t size = literalIds.size() ;
  theDerivatives.f = 0.0 ;
  theDerivatives.setDerivativesToZero() ;
  for (const auto& pair : expressions) {
    const bioDerivatives* condition = (pair.first)->getValueAndDerivatives(literalIds,false,false) ;
    if (condition->f != 0) {
      const bioDerivatives* fgh = (pair.second)->getValueAndDerivatives(literalIds,gradient,hessian) ;
      
      theDerivatives.f += fgh->f ;
      if (gradient) {
	    for (std::size_t k = 0 ; k < size ; ++k) {
	      theDerivatives.g[k] += fgh->g[k] ;
	      if (hessian) {
	        for (std::size_t l = k ; l < size ; ++l) {
	          theDerivatives.h[k][l] += fgh->h[k][l] ;
	        }
	      }
	    }
      }
    }
  }
  if (hessian) {
    // Fill the symmetric part of the matrix
    for (std::size_t k = 0 ; k < size ; ++k) {
      for (std::size_t l = k+1 ; l < size ; ++l) {
	    theDerivatives.h[l][k]  = theDerivatives.h[k][l] ;
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprConditionalSum::print(bioBoolean hp) const {
    std::stringstream str;
    const char* separator = hp ? ", " : " + ";
    const char* start = hp ? "ConditionalSum(" : "(";
    const char* end = ")";
    
    str << start;
    for (auto it = expressions.begin(); it != expressions.end(); ++it) {
        if (it != expressions.begin()) {
            str << separator;
        }
        str << it->first->print(hp) << ": " << it->second->print(hp);
    }
    str << end;
    return str.str();
}
