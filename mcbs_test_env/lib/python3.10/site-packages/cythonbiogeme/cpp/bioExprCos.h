//-*-c++-*----------------------%--------------------------------------
//
// File name : bioExprCos.h
// @date   Thu Sep 21 09:56:40 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprCos_h
#define bioExprCos_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprCos: public bioExpression {
public:
  bioExprCos(bioExpression* c) ;
  ~bioExprCos() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
  
protected:
  bioExpression* child ;
};
#endif
