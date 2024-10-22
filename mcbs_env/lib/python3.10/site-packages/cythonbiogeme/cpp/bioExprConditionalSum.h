//-*-c++-*------------------------------------------------------------
//
// File name : bioExprConditionalSum.h
// @date   Wed Sep 20 18:56:27 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprConditionalSum_h
#define bioExprConditionalSum_h

#include <unordered_map>
#include "bioExpression.h"
#include "bioString.h"

class bioExprConditionalSum: public bioExpression {
 public:
  bioExprConditionalSum(std::unordered_map<bioExpression*, bioExpression*> d) ;
  ~bioExprConditionalSum() ;
  
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) ;
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  std::unordered_map<bioExpression*, bioExpression*> expressions ;
};


#endif
