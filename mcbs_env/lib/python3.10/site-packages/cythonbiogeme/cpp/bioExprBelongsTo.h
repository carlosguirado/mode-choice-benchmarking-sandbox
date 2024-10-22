//-*-c++-*----------------------%--------------------------------------
//
// File name : bioExprBelongsTo.h
// @date   Tue Sep 19 14:07:49 2023
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprBelongsTo_h
#define bioExprBelongsTo_h

#include <set>
#include "bioExpression.h"
#include "bioString.h"

class bioExprBelongsTo: public bioExpression {
 public:
  bioExprBelongsTo(bioExpression* c, const std::set<bioReal>& the_set) ;
  ~bioExprBelongsTo() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;

 protected:
  bioExpression* child ;
  std::set<bioReal> the_set ;  
};
#endif
