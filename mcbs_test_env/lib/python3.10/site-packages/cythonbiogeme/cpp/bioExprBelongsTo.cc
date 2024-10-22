//-*-c++-*------------------------------------------------------------
//
// File name : bioExprBelongsTo.cc
// @date   Tue Sep 19 14:15:00 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include <sstream>
#include <cmath>
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioExprBelongsTo.h"

bioExprBelongsTo::bioExprBelongsTo(bioExpression* c, const std::set<bioReal>& the_set):
  child(c), the_set(the_set) {
  listOfChildren.push_back(c) ;
}

bioExprBelongsTo::~bioExprBelongsTo() {

}

const bioDerivatives* bioExprBelongsTo::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							       bioBoolean gradient,
							       bioBoolean hessian) {
  
  if (gradient || hessian) {
    throw bioExceptions(__FILE__,__LINE__,"This expression is not differentiable: " + print()) ;
  }
  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;
  theDerivatives.resize(literalIds.size()) ;
  
  const bioDerivatives* the_expression = child->getValueAndDerivatives(literalIds, false,false) ;

  bioReal value_found = static_cast<bioReal>(the_set.count(the_expression->f)) ;
  theDerivatives.f = value_found ;
  return &theDerivatives ;
}

bioString bioExprBelongsTo::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "BelongsTo[" << child->print(hp) << "](" ;
  for (auto it = the_set.begin(); it != the_set.end(); ++it) {
    str << *it;
    
    // If it's not the last element, append a comma and a space
    if (std::next(it) != the_set.end()) {
      str << ", ";
    }
  }
  str << ")" ;
  return str.str() ;
}
