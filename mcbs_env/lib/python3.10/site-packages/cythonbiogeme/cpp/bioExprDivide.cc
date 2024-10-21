//-*-c++-*------------------------------------------------------------
//
// File name : bioExprDivide.cc
// @date   Fri Apr 13 11:57:05 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprDivide.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioConstants.h"


bioExprDivide::bioExprDivide(bioExpression* l, bioExpression* r) :
  left(l), right(r) {
    listOfChildren.push_back(l) ;
    listOfChildren.push_back(r) ;

}

bioExprDivide::~bioExprDivide() {

}

const bioDerivatives* bioExprDivide::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						      bioBoolean gradient,
						      bioBoolean hessian) {

  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,__LINE__,"If the hessian is needed, the gradient must be computed") ;
  }



  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;
  

  const bioDerivatives* leftResult = left->getValueAndDerivatives(literalIds,gradient,hessian) ;
  const bioDerivatives* rightResult = right->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioReal rightValue = rightResult->f ;
  bioReal rSquare = rightValue * rightValue ;
  bioReal rCube = rSquare * rightValue ;

    // Numerator is zero
    if (leftResult-> f == 0) {
      theDerivatives.f = 0.0 ;
      if (gradient) {
        for (bioUInt i = 0 ; i < n ; ++i) {
          theDerivatives.g[i] = leftResult->g[i] / rightValue;
          if (hessian) {
            for (bioUInt j = i ; j < n ; ++j) {
              theDerivatives.h[i][j] =
              leftResult->h[i][j] / rightValue
              - leftResult->g[i] * rightResult->g[j] / rSquare
              - leftResult->g[j] * rightResult->g[i] / rSquare ;
              if (i != j) {
                theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
              }
            }
          }
        }
      }
      return &theDerivatives ;
    }
    // Denominator is 1.0
    if (rightValue == 1.0) {
      theDerivatives.f = leftResult->f ;
      if (gradient) {
        for (bioUInt i = 0 ; i < n ; ++i) {
          theDerivatives.g[i] = leftResult->g[i] - leftResult->f * rightResult->g[i] ;
          if (hessian) {
            for (bioUInt j = i ; j < n ; ++j) {
              theDerivatives.h[i][j] =
              leftResult->h[i][j]
              - leftResult->g[i] * rightResult->g[j]
              - leftResult->g[j] * rightResult->g[i]
              + 2.0 * leftResult->f * rightResult->g[i] * rightResult->g[j]
              - leftResult->f * rightResult->h[i][j] ;
              if (i != j) {
              theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
              }
            }
          }
        }
      }
      return &theDerivatives ;
    }
    // General case
    theDerivatives.f = leftResult->f / rightValue ;
    if (gradient) {
      for (bioUInt i = 0 ; i < n ; ++i) {
        theDerivatives.g[i] = 0.0 ;
        if (leftResult->g[i] != 0.0) {
          theDerivatives.g[i] += leftResult->g[i] / rightValue ;
        }
        if (rightResult->g[i] != 0.0) {
          theDerivatives.g[i] -= leftResult->f * rightResult->g[i] / rSquare ;
        }
        if (hessian) {
          for (bioUInt j = i ; j < n ; ++j) {
            theDerivatives.h[i][j] =
            leftResult->h[i][j] / rightValue
            - leftResult->g[i] * rightResult->g[j] / rSquare
            - leftResult->g[j] * rightResult->g[i] / rSquare
            + 2.0 * leftResult->f * rightResult->g[i] * rightResult->g[j] / rCube
            - leftResult->f * rightResult->h[i][j] / rSquare ;
            if (i != j) {
              theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
            }
          }
        }
      }

	return &theDerivatives ;
  }

  return &theDerivatives ;
}

bioString bioExprDivide::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "/(" << left->print(hp) << "," << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << "/" << right->print(hp) << ")" ;
  }
  return str.str() ;
}

