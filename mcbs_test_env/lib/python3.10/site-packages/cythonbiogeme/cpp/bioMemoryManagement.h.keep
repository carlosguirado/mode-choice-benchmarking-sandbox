//-*-c++-*------------------------------------------------------------
//
// File name : bioMemoryManagement.h
// @date   Sat Sep 26 12:19:14 2020
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioMemoryManagement_h
#define bioMemoryManagement_h

#include <vector>
#include <set>
#include <unordered_map>
#include "bioTypes.h"
#include "bioString.h"
#include "bioExprLinearUtility.h"

class bioExpression ;
class bioExprFreeParameter ;
class bioExprFixedParameter ;
class bioExprVariable ;
class bioExprDraws ;
class bioExprRandomVariable ;
class bioExprNumeric ;
class bioExprPlus ;
class bioExprMinus ;
class bioExprTimes ;
class bioExprDivide ;
class bioExprPower ;
class bioExprPowerConstant ;
class bioExprAnd ;
class bioExprOr ;
class bioExprEqual ;
class bioExprNotEqual ;
class bioExprLess ;
class bioExprLessOrEqual ;
class bioExprGreater ;
class bioExprGreaterOrEqual ;
class bioExprMin ;
class bioExprMax ;
class bioExprUnaryMinus ;
class bioExprMontecarlo ;
class bioExprNormalCdf ;
class bioExprPanelTrajectory ;
class bioExprExp ;
class bioExprSin ;
class bioExprCos ;
class bioExprLog ;
class bioExprLogzero ;
class bioExprBelongsTo ;
class bioExprDerive ;
class bioExprIntegrate ;
class bioExprLogLogit ;
class bioExprLogLogitFullChoiceSet ;
class bioExprMultSum ;
class bioExprConditionalSum ;
class bioExprElem ;

class bioSeveralExpressions ;

class bioMemoryManagement {

public:
  static bioMemoryManagement* the() ;
  ~bioMemoryManagement() ;
  void releaseAllMemory() ;
  bioExprFreeParameter* get_bioExprFreeParameter(bioUInt literalId,
						 bioUInt parameterId,
						 bioString name) ;
  bioExprFixedParameter* get_bioExprFixedParameter(bioUInt literalId,
						   bioUInt parameterId,
						   bioString name) ; 
  bioExprVariable* get_bioExprVariable(bioUInt literalId,
				       bioUInt variableId,
				       bioString name) ; 
  bioExprDraws* get_bioExprDraws(bioUInt uniqueId,
				 bioUInt drawId,
				 bioString name) ;
  bioExprRandomVariable* get_bioExprRandomVariable(bioUInt literalId,
						   bioUInt id,
						   bioString name) ;
  bioExprNumeric* get_bioExprNumeric(bioReal v) ;
  bioExprPlus* get_bioExprPlus(bioExpression* ell, bioExpression* r) ;
  bioExprMinus* get_bioExprMinus(bioExpression* ell, bioExpression* r) ;
  bioExprTimes* get_bioExprTimes(bioExpression* ell, bioExpression* r) ;
  bioExprDivide* get_bioExprDivide(bioExpression* ell, bioExpression* r) ;
  bioExprPower* get_bioExprPower(bioExpression* ell, bioExpression* r) ;
  bioExprPowerConstant* get_bioExprPowerConstant(bioExpression* ell, bioReal exponent) ;
  bioExprAnd* get_bioExprAnd(bioExpression* ell, bioExpression* r) ;
  bioExprOr* get_bioExprOr(bioExpression* ell, bioExpression* r) ;
  bioExprEqual* get_bioExprEqual(bioExpression* ell, bioExpression* r) ;
  bioExprNotEqual* get_bioExprNotEqual(bioExpression* ell, bioExpression* r) ;
  bioExprLess* get_bioExprLess(bioExpression* ell, bioExpression* r) ;
  bioExprLessOrEqual* get_bioExprLessOrEqual(bioExpression* ell, bioExpression* r) ;
  bioExprGreater* get_bioExprGreater(bioExpression* ell, bioExpression* r) ;
  bioExprGreaterOrEqual* get_bioExprGreaterOrEqual(bioExpression* ell, bioExpression* r) ;
  bioExprMin* get_bioExprMin(bioExpression* ell, bioExpression* r) ;
  bioExprMax* get_bioExprMax(bioExpression* ell, bioExpression* r) ;
  bioExprUnaryMinus* get_bioExprUnaryMinus(bioExpression* ell) ;
  bioExprMontecarlo* get_bioExprMontecarlo(bioExpression* ell) ;
  bioExprNormalCdf* get_bioExprNormalCdf(bioExpression* ell) ;
  bioExprPanelTrajectory* get_bioExprPanelTrajectory(bioExpression* ell) ;
  bioExprExp* get_bioExprExp(bioExpression* ell) ;
  bioExprSin* get_bioExprSin(bioExpression* ell) ;
  bioExprCos* get_bioExprCos(bioExpression* ell) ;
  bioExprLog* get_bioExprLog(bioExpression* ell) ;
  bioExprLogzero* get_bioExprLogzero(bioExpression* ell) ;
  bioExprDerive* get_bioExprDerive(bioExpression* c, bioUInt lid) ;
  bioExprBelongsTo* get_bioExprBelongsTo(bioExpression* c, const std::set<bioReal>& the_set) ;
  bioExprIntegrate* get_bioExprIntegrate(bioExpression* c, bioUInt lid) ;
  bioExprLinearUtility* get_bioExprLinearUtility(std::vector<bioLinearTerm> t) ;
  bioExprLogLogit* get_bioExprLogLogit(bioExpression* c,
				       std::map<bioUInt,bioExpression*> u,
				       std::map<bioUInt,bioExpression*> a) ;
  bioExprLogLogitFullChoiceSet* get_bioExprLogLogitFullChoiceSet(bioExpression* c,
								 std::map<bioUInt,bioExpression*> u) ;
  bioExprConditionalSum* get_bioExprConditionalSum(std::unordered_map<bioExpression*, bioExpression*> d) ;
  bioExprMultSum* get_bioExprMultSum(std::vector<bioExpression*> e) ;
  bioExprElem* get_bioExprElem(bioExpression* k, std::map<bioUInt,bioExpression*> d) ;
  bioSeveralExpressions* get_bioSeveralExpressions(std::vector<bioExpression*> exprs) ;
private:
  bioMemoryManagement() ;
  std::vector<bioExprFreeParameter*> a_bioExprFreeParameter ;
  std::vector<bioExprFixedParameter*> a_bioExprFixedParameter ;
  std::vector<bioExprVariable*> a_bioExprVariable ;
  std::vector<bioExprDraws*> a_bioExprDraws ;
  std::vector<bioExprRandomVariable*> a_bioExprRandomVariable ;
  std::vector<bioExprNumeric*> a_bioExprNumeric ;
  std::vector<bioExprPlus*> a_bioExprPlus ;
  std::vector<bioExprMinus*> a_bioExprMinus ;
  std::vector<bioExprTimes*> a_bioExprTimes ;
  std::vector<bioExprDivide*> a_bioExprDivide ;
  std::vector<bioExprPower*> a_bioExprPower ;
  std::vector<bioExprPowerConstant*> a_bioExprPowerConstant ;
  std::vector<bioExprAnd*> a_bioExprAnd ;
  std::vector<bioExprOr*> a_bioExprOr ;
  std::vector<bioExprEqual*> a_bioExprEqual ;
  std::vector<bioExprNotEqual*> a_bioExprNotEqual ;
  std::vector<bioExprLess*> a_bioExprLess ;
  std::vector<bioExprLessOrEqual*> a_bioExprLessOrEqual ;
  std::vector<bioExprGreater*> a_bioExprGreater ;
  std::vector<bioExprGreaterOrEqual*> a_bioExprGreaterOrEqual ;
  std::vector<bioExprMin*> a_bioExprMin ;
  std::vector<bioExprMax*> a_bioExprMax ;
  std::vector<bioExprUnaryMinus*> a_bioExprUnaryMinus ;
  std::vector<bioExprMontecarlo*> a_bioExprMontecarlo ;
  std::vector<bioExprNormalCdf*> a_bioExprNormalCdf ;
  std::vector<bioExprPanelTrajectory*> a_bioExprPanelTrajectory ;
  std::vector<bioExprExp*> a_bioExprExp ;
  std::vector<bioExprSin*> a_bioExprSin ;
  std::vector<bioExprCos*> a_bioExprCos ;
  std::vector<bioExprLog*> a_bioExprLog ;
  std::vector<bioExprLogzero*> a_bioExprLogzero ;
  std::vector<bioExprDerive*> a_bioExprDerive ;
  std::vector<bioExprBelongsTo*> a_bioExprBelongsTo ;
  std::vector<bioExprIntegrate*> a_bioExprIntegrate ;
  std::vector<bioExprLinearUtility*> a_bioExprLinearUtility ;
  std::vector<bioExprLogLogit*> a_bioExprLogLogit ;
  std::vector<bioExprLogLogitFullChoiceSet*> a_bioExprLogLogitFullChoiceSet ;
  std::vector<bioExprMultSum*> a_bioExprMultSum ;
  std::vector<bioExprConditionalSum*> a_bioExprConditionalSum ;
  std::vector<bioExprElem*> a_bioExprElem ;
  std::vector<bioSeveralExpressions*> a_bioSeveralExpressions ;
};
#endif
