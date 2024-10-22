//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLogLogit.cc
// @date   Fri Apr 13 15:16:24 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include <sstream>
#include <cmath>

#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioExprLogLogit.h"
#include "bioConstants.h"


bioExprLogLogit::bioExprLogLogit(bioExpression* c,
				 std::map<bioUInt,bioExpression*> u,
				 std::map<bioUInt,bioExpression*> a) :
  choice(c), utilities(u), availabilities(a), expi(u.size()) {
  listOfChildren.push_back(choice) ;
  for (std::map<bioUInt,bioExpression*>::iterator i = u.begin() ;
       i != u.end();
       ++i) {
    listOfChildren.push_back(i->second) ;
  }
  for (std::map<bioUInt,bioExpression*>::iterator i = a.begin() ;
       i != a.end();
       ++i) {
    listOfChildren.push_back(i->second) ;
  }
  Vs.reserve(utilities.size()) ;
}

bioExprLogLogit::~bioExprLogLogit() {

}

const bioDerivatives* bioExprLogLogit::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							bioBoolean gradient,
							bioBoolean hessian) {

  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,
			__LINE__,
			"If the hessian is needed, the gradient must be computed") ;
  }
  
  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;
  weightedSum.resize(n) ;

  chosen = bioUInt(choice->getValue()) ;
  const bioDerivatives* chosenUtility(NULL) ;
  largestUtility = -bioMaxReal ;
  Vs.clear();


  for (std::map<bioUInt,bioExpression*>::iterator avail = availabilities.begin() ;
       avail != availabilities.end() ;
       ++avail) {
  //for (const auto& avail : availabilities) {
    av = avail->second->getValue() ;
    if (av == 0.0) {
      if (avail->first == chosen) {
        // The chosen alternative is not available
	    theDerivatives.setDerivativesToZero() ;
	    theDerivatives.f = log(0.0) ;
        return &theDerivatives ;
      }
    }
    else {
      //std::map<bioUInt,bioExpression*>::iterator theUtil = utilities.find(i->first) ;

      theUtil = utilities.find(avail->first) ;
      if (theUtil == utilities.end()) {
	    std::stringstream str ;
	    str << "Inconsistent dictionaries. Alternative " << avail->first << " defined in the availabilities, and not in the utilities" ;
	    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
      }
      if (theUtil->second == NULL) {
	    throw bioExceptNullPointer(__FILE__,__LINE__,"formula") ;
      }
      const bioDerivatives* V = theUtil->second->getValueAndDerivatives(literalIds,gradient,hessian) ;
      if (V == NULL) {
	    throw bioExceptNullPointer(__FILE__,__LINE__,"result") ;
      }
      DEBUG_MESSAGE("V[" << avail->first << "]=" << V->f) ;
      if (V->f > largestUtility) {
        largestUtility = V->f ;
      }
      if (avail->first == chosen) {
	    chosenUtility = V ;
      }
      Vs.push_back(*V) ;
    }
  }


  if (chosenUtility == NULL) {
    std::stringstream str ;
    str << "Alternative "
	<< chosen
	<< " is not known. The alternatives that have been defined are" ;
    for (std::map<bioUInt,bioExpression*>::iterator i = utilities.begin() ;
	 i != utilities.end() ;
	 ++i) {
      str << " " << i->first ;
    }
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  
  shift = ceil(largestUtility / 10.0) * 10.0 ;



  denominator = 0.0 ;
  for (bioUInt k = 0 ; k < Vs.size() ; ++k) {
    expi[k] = exp(Vs[k].f - shift) ;
    denominator += expi[k] ;
  }

  theDerivatives.f = chosenUtility->f - log(denominator) - shift ;

  DEBUG_MESSAGE("Logit = " << chosenUtility->f  << " - " << log(denominator) <<  " - " << shift << " = " << theDerivatives.f) ;

  if (gradient) {

    std::fill(weightedSum.begin(), weightedSum.end(), 0.0);
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt k = 0 ; k < Vs.size() ; ++k) {
	    if (Vs[k].g[i] != 0.0) {
	      weightedSum[i] += Vs[k].g[i] * expi[k] ;
	    }
      }
      theDerivatives.g[i] = chosenUtility->g[i] ;
      if (weightedSum[i] != 0.0) {
	    theDerivatives.g[i] -= weightedSum[i] / denominator ;
      }
    }
    if (hessian) {
      dsquare = denominator * denominator ;
      for (bioUInt i = 0 ; i < n ; ++i) {
	    for (bioUInt j = i ; j < n ; ++j) {
	      dsecond = 0.0 ;
	      for (bioUInt k = 0 ; k < Vs.size() ; ++k ) {
	        the_term = 0.0 ;
	        if (Vs[k].g[i] != 0 && Vs[k].g[j] != 0.0) {
	          the_term += Vs[k].g[i] * Vs[k].g[j] ;
	        }
	        vih = Vs[k].h[i][j] ;
	        if (vih != 0.0) {
	          the_term += vih ;
	        }
	        the_term *= expi[k] ;
	        dsecond += the_term ;
	      }
	      v =  chosenUtility->h[i][j] ;
	      v1 = 0.0 ;
	      if (weightedSum[i] != 0.0 && weightedSum[j] != 0.0) {
	        v1 = weightedSum[i] * weightedSum[j] / dsquare ;
	      }
	      v2 =  dsecond / denominator ;
	      theDerivatives.h[i][j] = v+v1-v2 ;
	      if ( i != j) {
	        theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
	      }
	    }
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprLogLogit::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "Logit[" << choice->print(hp) << "](" ;
  for (std::map<bioUInt,bioExpression*>::const_iterator i = utilities.begin() ;
       i != utilities.end() ;
       ++i) {
    if (i != utilities.begin()) {
      str << ";" ;
    }
    str << "{" << availabilities.at(i->first)->print(hp) << "}" << i->second->print(hp) ;
  }
  str << ")" ;
  return str.str() ;
}
