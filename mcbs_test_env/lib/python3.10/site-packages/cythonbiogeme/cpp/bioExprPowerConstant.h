//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPowerConstant.h
// Date: Wed Jul 3 10:19:59 2024
// Author: Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprPowerConstant_h
#define bioExprPowerConstant_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprPowerConstant: public bioExpression {
 public:
  bioExprPowerConstant(bioExpression* l, bioReal exponent) ;
  ~bioExprPowerConstant() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioExpression* child ;
  bioReal exponent ;
};
#endif
