//-*-c++-*------------------------------------------------------------
//
// File name : bioMemoryManagement.cc
// @date   Sat Sep 26 12:22:24 2020
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioMemoryManagement.h"
#include "bioDebug.h"
#include "bioExpression.h"
#include "bioExprFreeParameter.h"
#include "bioExprFixedParameter.h"
#include "bioExprVariable.h"
#include "bioExprDraws.h"
#include "bioExprRandomVariable.h"
#include "bioExprNumeric.h"
#include "bioExprPlus.h"
#include "bioExprMinus.h"
#include "bioExprTimes.h"
#include "bioExprDivide.h"
#include "bioExprPower.h"
#include "bioExprPowerConstant.h"
#include "bioExprAnd.h"
#include "bioExprOr.h"
#include "bioExprEqual.h"
#include "bioExprNotEqual.h"
#include "bioExprLess.h"
#include "bioExprLessOrEqual.h"
#include "bioExprGreater.h"
#include "bioExprGreaterOrEqual.h"
#include "bioExprMin.h"
#include "bioExprMax.h"
#include "bioExprUnaryMinus.h"
#include "bioExprMontecarlo.h"
#include "bioExprNormalCdf.h"
#include "bioExprPanelTrajectory.h"
#include "bioExprExp.h"
#include "bioExprSin.h"
#include "bioExprCos.h"
#include "bioExprLog.h"
#include "bioExprLogzero.h"
#include "bioExprBelongsTo.h"
#include "bioExprDerive.h"
#include "bioExprIntegrate.h"
#include "bioExprLogLogit.h"
#include "bioExprLogLogitFullChoiceSet.h"
#include "bioExprMultSum.h"
#include "bioExprConditionalSum.h"
#include "bioExprElem.h"
#include "bioSeveralExpressions.h"

bioMemoryManagement::bioMemoryManagement() {

}

bioMemoryManagement::~bioMemoryManagement() {
  releaseAllMemory() ;
}


void bioMemoryManagement::releaseAllMemory() {
 // for (std::vector<bioExprFreeParameter*>::iterator i = a_bioExprFreeParameter.begin() ;
 //      i != a_bioExprFreeParameter.end() ;
 //      ++i) {
 //   delete(*i) ;
 // }
  a_bioExprFreeParameter.clear() ;


 // for (std::vector<bioExprFixedParameter*>::iterator i = a_bioExprFixedParameter.begin() ;
 //      i != a_bioExprFixedParameter.end() ;
 //      ++i) {
 //   delete(*i) ;
 // }
  a_bioExprFixedParameter.clear() ;

  //for (std::vector<bioExprFixedParameter*>::iterator i = a_bioExprFixedParameter.begin() ;
  //     i != a_bioExprFixedParameter.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprFixedParameter.clear() ;

  //for (std::vector<bioExprVariable*>::iterator i = a_bioExprVariable.begin() ;
  //     i != a_bioExprVariable.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprVariable.clear() ;

  //for (std::vector<bioExprDraws*>::iterator i = a_bioExprDraws.begin() ;
  //     i != a_bioExprDraws.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprDraws.clear() ;

  //for (std::vector<bioExprRandomVariable*>::iterator i = a_bioExprRandomVariable.begin() ;
  //     i != a_bioExprRandomVariable.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprRandomVariable.clear() ;
  //for (std::vector<bioExprNumeric*>::iterator i = a_bioExprNumeric.begin() ;
  //     i != a_bioExprNumeric.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprNumeric.clear() ;
  //for (std::vector<bioExprPlus*>::iterator i = a_bioExprPlus.begin() ;
  //     i != a_bioExprPlus.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprPlus.clear() ;
  //for (std::vector<bioExprMinus*>::iterator i = a_bioExprMinus.begin() ;
  //     i != a_bioExprMinus.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprMinus.clear() ;
  //for (std::vector<bioExprTimes*>::iterator i = a_bioExprTimes.begin() ;
  //     i != a_bioExprTimes.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprTimes.clear() ;
  //for (std::vector<bioExprDivide*>::iterator i = a_bioExprDivide.begin() ;
  //     i != a_bioExprDivide.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprDivide.clear() ;
  //for (std::vector<bioExprPower*>::iterator i = a_bioExprPower.begin() ;
  //     i != a_bioExprPower.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprPower.clear() ;
  //for (std::vector<bioExprPowerConstant*>::iterator i = a_bioExprPowerConstant.begin() ;
  //     i != a_bioExprPowerConstant.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprPowerConstant.clear() ;
  //for (std::vector<bioExprAnd*>::iterator i = a_bioExprAnd.begin() ;
  //     i != a_bioExprAnd.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprAnd.clear() ;
  //for (std::vector<bioExprOr*>::iterator i = a_bioExprOr.begin() ;
  //     i != a_bioExprOr.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprOr.clear() ;
  //for (std::vector<bioExprEqual*>::iterator i = a_bioExprEqual.begin() ;
  //     i != a_bioExprEqual.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprEqual.clear() ;
  //for (std::vector<bioExprNotEqual*>::iterator i = a_bioExprNotEqual.begin() ;
  //     i != a_bioExprNotEqual.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprNotEqual.clear() ;
  //for (std::vector<bioExprLess*>::iterator i = a_bioExprLess.begin() ;
  //     i != a_bioExprLess.end() ;
  //     ++i) {
  // delete(*i) ;
  //}
  a_bioExprLess.clear() ;
  //for (std::vector<bioExprLessOrEqual*>::iterator i = a_bioExprLessOrEqual.begin() ;
  //     i != a_bioExprLessOrEqual.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprLessOrEqual.clear() ;
  //for (std::vector<bioExprGreater*>::iterator i = a_bioExprGreater.begin() ;
  //     i != a_bioExprGreater.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprGreater.clear() ;
  //for (std::vector<bioExprGreaterOrEqual*>::iterator i = a_bioExprGreaterOrEqual.begin() ;
  //     i != a_bioExprGreaterOrEqual.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprGreaterOrEqual.clear() ;
  //for (std::vector<bioExprMin*>::iterator i = a_bioExprMin.begin() ;
  //     i != a_bioExprMin.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprMin.clear() ;
  //for (std::vector<bioExprMax*>::iterator i = a_bioExprMax.begin() ;
  //     i != a_bioExprMax.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprMax.clear() ;
  //for (std::vector<bioExprUnaryMinus*>::iterator i = a_bioExprUnaryMinus.begin() ;
  //     i != a_bioExprUnaryMinus.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprUnaryMinus.clear() ;
  //for (std::vector<bioExprMontecarlo*>::iterator i = a_bioExprMontecarlo.begin() ;
  //     i != a_bioExprMontecarlo.end() ;
  //    ++i) {
  //  delete(*i) ;
  //}
  a_bioExprMontecarlo.clear() ;
  //for (std::vector<bioExprNormalCdf*>::iterator i = a_bioExprNormalCdf.begin() ;
  //     i != a_bioExprNormalCdf.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprNormalCdf.clear() ;
  //for (std::vector<bioExprPanelTrajectory*>::iterator i = a_bioExprPanelTrajectory.begin() ;
  //     i != a_bioExprPanelTrajectory.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprPanelTrajectory.clear() ;
  //for (std::vector<bioExprExp*>::iterator i = a_bioExprExp.begin() ;
  //     i != a_bioExprExp.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprExp.clear() ;
  //for (std::vector<bioExprSin*>::iterator i = a_bioExprSin.begin() ;
  //     i != a_bioExprSin.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprSin.clear() ;
  //for (std::vector<bioExprCos*>::iterator i = a_bioExprCos.begin() ;
  //     i != a_bioExprCos.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprCos.clear() ;
  //for (std::vector<bioExprLog*>::iterator i = a_bioExprLog.begin() ;
  //     i != a_bioExprLog.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprLog.clear() ;
  //for (std::vector<bioExprLogzero*>::iterator i = a_bioExprLogzero.begin() ;
  //     i != a_bioExprLogzero.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprLogzero.clear() ;
  //for (std::vector<bioExprBelongsTo*>::iterator i = a_bioExprBelongsTo.begin() ;
  //     i != a_bioExprBelongsTo.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprBelongsTo.clear() ;
  //for (std::vector<bioExprDerive*>::iterator i = a_bioExprDerive.begin() ;
  //     i != a_bioExprDerive.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprDerive.clear() ;
  //for (std::vector<bioExprIntegrate*>::iterator i = a_bioExprIntegrate.begin() ;
  //     i != a_bioExprIntegrate.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprIntegrate.clear() ;
  //for (std::vector<bioExprLinearUtility*>::iterator i = a_bioExprLinearUtility.begin() ;
  //     i != a_bioExprLinearUtility.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprLinearUtility.clear() ;
  //for (std::vector<bioExprLogLogit*>::iterator i = a_bioExprLogLogit.begin() ;
  //     i != a_bioExprLogLogit.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprLogLogit.clear() ;
  //for (std::vector<bioExprLogLogitFullChoiceSet*>::iterator i = a_bioExprLogLogitFullChoiceSet.begin() ;
  //     i != a_bioExprLogLogitFullChoiceSet.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprLogLogitFullChoiceSet.clear() ;
  //for (std::vector<bioExprMultSum*>::iterator i = a_bioExprMultSum.begin() ;
  //     i != a_bioExprMultSum.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprMultSum.clear() ;
  //for (std::vector<bioExprConditionalSum*>::iterator i = a_bioExprConditionalSum.begin() ;
  //     i != a_bioExprConditionalSum.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprConditionalSum.clear() ;
  //for (std::vector<bioExprElem*>::iterator i = a_bioExprElem.begin() ;
  //     i != a_bioExprElem.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioExprElem.clear() ;

  //for (std::vector<bioSeveralExpressions*>::iterator i = a_bioSeveralExpressions.begin() ;
  //     i != a_bioSeveralExpressions.end() ;
  //     ++i) {
  //  delete(*i) ;
  //}
  a_bioSeveralExpressions.clear() ;

}

bioMemoryManagement* bioMemoryManagement::the() {
  static bioMemoryManagement* singleInstance = NULL;
  if (singleInstance == NULL) {
    singleInstance = new bioMemoryManagement() ;
  } 
  return singleInstance ;
  
}

bioExprFreeParameter* bioMemoryManagement::get_bioExprFreeParameter(bioUInt literalId,
								     bioUInt parameterId,
								     bioString name) {

  std::unique_ptr<bioExprFreeParameter> ptr = std::make_unique<bioExprFreeParameter>(literalId,
						       parameterId,
						       name) ;
  bioExprFreeParameter* rawPtr = ptr.get() ;
  a_bioExprFreeParameter.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprFixedParameter* bioMemoryManagement::get_bioExprFixedParameter(bioUInt literalId,
								      bioUInt parameterId,
								      bioString name) {
  std::unique_ptr<bioExprFixedParameter> ptr = std::make_unique<bioExprFixedParameter>(literalId,
							 parameterId,
							 name) ;
  bioExprFixedParameter* rawPtr = ptr.get() ;
  a_bioExprFixedParameter.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprVariable* bioMemoryManagement::get_bioExprVariable(bioUInt literalId,
						      bioUInt variableId,
						      bioString name) {
  std::unique_ptr<bioExprVariable> ptr = std::make_unique<bioExprVariable>(literalId,
					     variableId,
					     name) ;
  bioExprVariable* rawPtr = ptr.get() ;
  a_bioExprVariable.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprDraws* bioMemoryManagement::get_bioExprDraws(bioUInt uniqueId,
					     bioUInt drawId,
					     bioString name) {
  std::unique_ptr<bioExprDraws> ptr = std::make_unique<bioExprDraws>(uniqueId,
			 drawId,
			 name) ;
  bioExprDraws* rawPtr = ptr.get() ;
  a_bioExprDraws.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprRandomVariable* bioMemoryManagement::get_bioExprRandomVariable(bioUInt literalId,
									bioUInt id,
									bioString name) {
  std::unique_ptr<bioExprRandomVariable> ptr = std::make_unique<bioExprRandomVariable>(literalId,
							 id,
							 name) ;
  bioExprRandomVariable* rawPtr = ptr.get() ;
  a_bioExprRandomVariable.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprNumeric* bioMemoryManagement::get_bioExprNumeric(bioReal v) {
  std::unique_ptr<bioExprNumeric> ptr = std::make_unique<bioExprNumeric>(v) ;
  bioExprNumeric* rawPtr = ptr.get() ;
  a_bioExprNumeric.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprPlus* bioMemoryManagement::get_bioExprPlus(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprPlus> ptr = std::make_unique<bioExprPlus>(ell, r) ;
  bioExprPlus* rawPtr = ptr.get() ;
  a_bioExprPlus.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprMinus* bioMemoryManagement::get_bioExprMinus(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprMinus> ptr = std::make_unique<bioExprMinus>(ell, r) ;
  bioExprMinus* rawPtr = ptr.get() ;
  a_bioExprMinus.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprTimes* bioMemoryManagement::get_bioExprTimes(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprTimes> ptr = std::make_unique<bioExprTimes>(ell, r) ;
  bioExprTimes* rawPtr = ptr.get() ;
  a_bioExprTimes.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprDivide* bioMemoryManagement::get_bioExprDivide(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprDivide> ptr = std::make_unique<bioExprDivide>(ell, r) ;
  bioExprDivide* rawPtr = ptr.get() ;
  a_bioExprDivide.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprPower* bioMemoryManagement::get_bioExprPower(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprPower> ptr = std::make_unique<bioExprPower>(ell, r) ;
  bioExprPower* rawPtr = ptr.get() ;
  a_bioExprPower.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprPowerConstant* bioMemoryManagement::get_bioExprPowerConstant(bioExpression* ell, bioReal exponent) {
  std::unique_ptr<bioExprPowerConstant> ptr = std::make_unique<bioExprPowerConstant>(ell, exponent) ;
  bioExprPowerConstant* rawPtr = ptr.get() ;
  a_bioExprPowerConstant.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprAnd* bioMemoryManagement::get_bioExprAnd(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprAnd> ptr = std::make_unique<bioExprAnd>(ell, r) ;
  bioExprAnd* rawPtr = ptr.get() ;
  a_bioExprAnd.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprOr* bioMemoryManagement::get_bioExprOr(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprOr> ptr = std::make_unique<bioExprOr>(ell, r) ;
  bioExprOr* rawPtr = ptr.get()  ;
  a_bioExprOr.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprEqual* bioMemoryManagement::get_bioExprEqual(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprEqual> ptr = std::make_unique<bioExprEqual>(ell, r) ;
  bioExprEqual* rawPtr = ptr.get() ;
  a_bioExprEqual.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprNotEqual* bioMemoryManagement::get_bioExprNotEqual(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprNotEqual> ptr = std::make_unique<bioExprNotEqual>(ell, r) ;
  bioExprNotEqual* rawPtr = ptr.get() ;
  a_bioExprNotEqual.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprLess* bioMemoryManagement::get_bioExprLess(bioExpression* ell, bioExpression* r) {
  std::unique_ptr<bioExprLess> ptr = std::make_unique<bioExprLess>(ell, r) ;
  bioExprLess* rawPtr = ptr.get() ;
  a_bioExprLess.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprLessOrEqual* bioMemoryManagement::get_bioExprLessOrEqual(bioExpression* ell,
								bioExpression* r) {
  std::unique_ptr<bioExprLessOrEqual> ptr = std::make_unique<bioExprLessOrEqual>(ell, r) ;
  bioExprLessOrEqual* rawPtr = ptr.get() ;
  a_bioExprLessOrEqual.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprGreater* bioMemoryManagement::get_bioExprGreater(bioExpression* ell,
							bioExpression* r) {
  std::unique_ptr<bioExprGreater> ptr = std::make_unique<bioExprGreater>(ell, r) ;
  bioExprGreater* rawPtr = ptr.get() ;
  a_bioExprGreater.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprGreaterOrEqual* bioMemoryManagement::get_bioExprGreaterOrEqual(bioExpression* ell,
								      bioExpression* r) {
  std::unique_ptr<bioExprGreaterOrEqual> ptr = std::make_unique<bioExprGreaterOrEqual>(ell, r) ;
  bioExprGreaterOrEqual* rawPtr = ptr.get() ;
  a_bioExprGreaterOrEqual.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprMin* bioMemoryManagement::get_bioExprMin(bioExpression* ell,
						bioExpression* r) {
  std::unique_ptr<bioExprMin> ptr = std::make_unique<bioExprMin>(ell, r) ;
  bioExprMin* rawPtr = ptr.get() ;
  a_bioExprMin.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprMax* bioMemoryManagement::get_bioExprMax(bioExpression* ell,
						bioExpression* r) {
  std::unique_ptr<bioExprMax> ptr = std::make_unique<bioExprMax>(ell, r) ;
  bioExprMax* rawPtr = ptr.get() ;
  a_bioExprMax.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprUnaryMinus* bioMemoryManagement::get_bioExprUnaryMinus(bioExpression* c) {
  std::unique_ptr<bioExprUnaryMinus> ptr = std::make_unique<bioExprUnaryMinus>(c) ;
  bioExprUnaryMinus* rawPtr = ptr.get() ;
  a_bioExprUnaryMinus.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprMontecarlo* bioMemoryManagement::get_bioExprMontecarlo(bioExpression* c) {
  std::unique_ptr<bioExprMontecarlo> ptr = std::make_unique<bioExprMontecarlo>(c) ;
  bioExprMontecarlo* rawPtr = ptr.get() ;
  a_bioExprMontecarlo.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprNormalCdf* bioMemoryManagement::get_bioExprNormalCdf(bioExpression* c) {
  std::unique_ptr<bioExprNormalCdf> ptr = std::make_unique<bioExprNormalCdf>(c) ;
  bioExprNormalCdf* rawPtr = ptr.get() ;
  a_bioExprNormalCdf.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprPanelTrajectory* bioMemoryManagement::get_bioExprPanelTrajectory(bioExpression* c) {
  std::unique_ptr<bioExprPanelTrajectory> ptr = std::make_unique<bioExprPanelTrajectory>(c) ;
  bioExprPanelTrajectory* rawPtr = ptr.get() ;
  a_bioExprPanelTrajectory.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprExp* bioMemoryManagement::get_bioExprExp(bioExpression* c) {
  std::unique_ptr<bioExprExp> ptr = std::make_unique<bioExprExp>(c) ;
  bioExprExp* rawPtr = ptr.get() ;
  a_bioExprExp.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprSin* bioMemoryManagement::get_bioExprSin(bioExpression* c) {
  std::unique_ptr<bioExprSin> ptr = std::make_unique<bioExprSin>(c) ;
  bioExprSin* rawPtr = ptr.get() ;
  a_bioExprSin.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprCos* bioMemoryManagement::get_bioExprCos(bioExpression* c) {
  std::unique_ptr<bioExprCos> ptr = std::make_unique<bioExprCos>(c) ;
  bioExprCos* rawPtr = ptr.get() ;
  a_bioExprCos.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprLog* bioMemoryManagement::get_bioExprLog(bioExpression* c) {
  std::unique_ptr<bioExprLog> ptr = std::make_unique<bioExprLog>(c) ;
  bioExprLog* rawPtr = ptr.get() ;
  a_bioExprLog.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprLogzero* bioMemoryManagement::get_bioExprLogzero(bioExpression* c) {
  std::unique_ptr<bioExprLogzero> ptr = std::make_unique<bioExprLogzero>(c) ;
  bioExprLogzero* rawPtr = ptr.get() ;
  a_bioExprLogzero.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprBelongsTo* bioMemoryManagement::get_bioExprBelongsTo(bioExpression* c, const std::set<bioReal>& the_set) {
  std::unique_ptr<bioExprBelongsTo> ptr = std::make_unique<bioExprBelongsTo>(c, the_set) ;
  bioExprBelongsTo* rawPtr = ptr.get() ;
  a_bioExprBelongsTo.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprDerive* bioMemoryManagement::get_bioExprDerive(bioExpression* c, bioUInt lid) {
  std::unique_ptr<bioExprDerive> ptr = std::make_unique<bioExprDerive>(c, lid) ;
  bioExprDerive* rawPtr = ptr.get() ;
  a_bioExprDerive.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprIntegrate* bioMemoryManagement::get_bioExprIntegrate(bioExpression* c, bioUInt lid) {
  std::unique_ptr<bioExprIntegrate> ptr = std::make_unique<bioExprIntegrate>(c, lid) ;
  bioExprIntegrate* rawPtr = ptr.get() ;
  a_bioExprIntegrate.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprLinearUtility* bioMemoryManagement::get_bioExprLinearUtility(std::vector<bioLinearTerm> t) {
  std::unique_ptr<bioExprLinearUtility> ptr = std::make_unique<bioExprLinearUtility>(t) ;
  bioExprLinearUtility* rawPtr = ptr.get() ;
  a_bioExprLinearUtility.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprLogLogit* bioMemoryManagement::get_bioExprLogLogit(bioExpression* c,
							  std::map<bioUInt,bioExpression*> u,
							  std::map<bioUInt,bioExpression*> a) {
  std::unique_ptr<bioExprLogLogit> ptr = std::make_unique<bioExprLogLogit>(c, u, a) ;
  bioExprLogLogit* rawPtr = ptr.get() ;
  a_bioExprLogLogit.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprLogLogitFullChoiceSet* bioMemoryManagement::get_bioExprLogLogitFullChoiceSet(bioExpression* c,
							  std::map<bioUInt,bioExpression*> u) {
  std::unique_ptr<bioExprLogLogitFullChoiceSet> ptr = std::make_unique<bioExprLogLogitFullChoiceSet>(c, u) ;
  bioExprLogLogitFullChoiceSet* rawPtr = ptr.get() ;
  a_bioExprLogLogitFullChoiceSet.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprMultSum* bioMemoryManagement::get_bioExprMultSum(std::vector<bioExpression*> e) {
  std::unique_ptr<bioExprMultSum> ptr = std::make_unique<bioExprMultSum>(e) ;
  bioExprMultSum* rawPtr = ptr.get() ;
  a_bioExprMultSum.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprConditionalSum* bioMemoryManagement::get_bioExprConditionalSum(std::unordered_map<bioExpression*, bioExpression*> d) {
  std::unique_ptr<bioExprConditionalSum> ptr = std::make_unique<bioExprConditionalSum>(d) ;
  bioExprConditionalSum* rawPtr = ptr.get() ;
  a_bioExprConditionalSum.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioExprElem* bioMemoryManagement::get_bioExprElem(bioExpression* k,
						  std::map<bioUInt,bioExpression*> d) {
  std::unique_ptr<bioExprElem> ptr = std::make_unique<bioExprElem>(k, d) ;
  bioExprElem* rawPtr = ptr.get() ;
  a_bioExprElem.push_back(std::move(ptr)) ;
  return rawPtr ;
}

bioSeveralExpressions* bioMemoryManagement::get_bioSeveralExpressions(std::vector<bioExpression*> exprs) {
  std::unique_ptr<bioSeveralExpressions> ptr = std::make_unique<bioSeveralExpressions>(exprs) ;
  bioSeveralExpressions* rawPtr = ptr.get() ;
  a_bioSeveralExpressions.push_back(std::move(ptr)) ;
  return rawPtr ;
}


