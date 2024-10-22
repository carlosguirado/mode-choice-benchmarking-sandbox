//-*-c++-*----------------------%--------------------------------------
//
// File name : bioExprSin.h
// @date   Thu Sep 21 09:56:16 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprSin_h
#define bioExprSin_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprSin: public bioExpression {
public:
  bioExprSin(bioExpression* c) ;
  ~bioExprSin() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
  
protected:
  bioExpression* child ;
};
#endif
