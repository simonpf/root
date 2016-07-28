// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh 18/07/16

/*************************************************************************
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Definition of the Timings struct that is used to hold timing //
// results from profiling runs of the DNN.                      //
//////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_TIMINGS
#define TMVA_DNN_TIMINGS

#include <iostream>
#include "Rtypes.h"

namespace TMVA {
namespace DNN {

struct TTimings
{
    // Propagation.
    Double_t TimeMultiplyTranspose = 0.0;
    Double_t TimeAddRowWise        = 0.0;
    Double_t TimeBackward          = 0.0;

    // Activation Functions.
    Double_t TimeIdentity                = 0.0;
    Double_t TimeIdentityDerivative      = 0.0;
    Double_t TimeRelu                    = 0.0;
    Double_t TimeReluDerivative          = 0.0;
    Double_t TimeSigmoid                 = 0.0;
    Double_t TimeSigmoidDerivative       = 0.0;
    Double_t TimeTanh                    = 0.0;
    Double_t TimeTanhDerivative          = 0.0;
    Double_t TimeSymmetricRelu           = 0.0;
    Double_t TimeSymmetricReluDerivative = 0.0;
    Double_t TimeSoftSign                = 0.0;
    Double_t TimeSoftSignDerivative      = 0.0;
    Double_t TimeGauss                   = 0.0;
    Double_t TimeGaussDerivative         = 0.0;

    // Loss Functions.
    Double_t TimeMeanSquaredError          = 0.0;
    Double_t TimeMeanSquaredErrorGradients = 0.0;
    Double_t TimeCrossEntropy              = 0.0;
    Double_t TimeCrossEntropyGradients     = 0.0;

    // Output Functions.
    Double_t TimeSigmoidOutput = 0.0;

    //Regularization
    Double_t TimeL1Regularization             = 0.0;
    Double_t TimeAddL1RegularizationGradients = 0.0;
    Double_t TimeL2Regularization             = 0.0;
    Double_t TimeAddL2RegularizationGradients = 0.0;

    // Dropout
    Double_t TimeDropout = 0.0;

    // Synchronization
    Double_t TimeSynchronization = 0.0;

    void Reset();
    void Print() const;
};

inline void TTimings::Reset()
{
    // Propagation.
    TimeMultiplyTranspose = 0.0;
    TimeAddRowWise        = 0.0;
    TimeBackward          = 0.0;

    // Activation Functions.
    TimeIdentity                = 0.0;
    TimeIdentityDerivative      = 0.0;
    TimeRelu                    = 0.0;
    TimeReluDerivative          = 0.0;
    TimeSigmoid                 = 0.0;
    TimeSigmoidDerivative       = 0.0;
    TimeTanh                    = 0.0;
    TimeTanhDerivative          = 0.0;
    TimeSymmetricRelu           = 0.0;
    TimeSymmetricReluDerivative = 0.0;
    TimeSoftSign                = 0.0;
    TimeSoftSignDerivative      = 0.0;
    TimeGauss                   = 0.0;
    TimeGaussDerivative         = 0.0;

    // Loss Functions.
    TimeMeanSquaredError          = 0.0;
    TimeMeanSquaredErrorGradients = 0.0;
    TimeCrossEntropy              = 0.0;
    TimeCrossEntropyGradients     = 0.0;

    // Output Functions.
    TimeSigmoidOutput = 0.0;

    //Regularization
    TimeL1Regularization             = 0.0;
    TimeAddL1RegularizationGradients = 0.0;
    TimeL2Regularization             = 0.0;
    TimeAddL2RegularizationGradients = 0.0;

    // Dropout
    TimeDropout = 0.0;
};

inline void TTimings::Print() const
{
    std::cout << "=== DNN Timing Results === [ms]" << std::endl << std::endl;

    // Propagation.
    if (TimeMultiplyTranspose != 0.0)
        std::cout << "MultiplyTranspose: " << TimeMultiplyTranspose << std::endl;
    if (TimeAddRowWise != 0.0)
        std::cout << "AddRowWise:        " << TimeAddRowWise << std::endl;
    if (TimeBackward != 0.0)
        std::cout << "Backward:          " << TimeBackward << std::endl;
    std::cout << std::endl;

    // Activation Functions.
    if (TimeIdentity != 0.0)
        std::cout << "Identity:                " << TimeIdentity << std::endl;
    if (TimeIdentityDerivative != 0.0)
        std::cout << "Time IdentityDerivative: " << TimeIdentityDerivative
                  << std::endl;
    if (TimeRelu != 0.0)
        std::cout << "Relu:                    " << TimeRelu << std::endl;
    if (TimeReluDerivative != 0.0)
        std::cout << "Time Relu Derivative:    " << TimeReluDerivative << std::endl;
    if (TimeSigmoid != 0.0)
        std::cout << "Sigmoid :                " << TimeSigmoid << std::endl;
    if (TimeSigmoidDerivative != 0.0)
        std::cout << "SigmoidDerivative:       " << TimeSigmoidDerivative
                  << std::endl;
    if (TimeTanh != 0.0)
        std::cout << "Tanh:                    " << TimeTanh << std::endl;
    if (TimeTanhDerivative != 0.0)
        std::cout << "TanhDerivative:          " << TimeTanhDerivative << std::endl;
    if (TimeSymmetricRelu != 0.0) {
        std::cout << "SymmetricRelu:           " << TimeSymmetricRelu << std::endl;
    }
    if (TimeSymmetricReluDerivative != 0.0) {
        std::cout << "SymmetricReluDerivative: " << TimeSymmetricReluDerivative
                  << std::endl;
    }
    if (TimeSoftSign != 0.0)
        std::cout << "SoftSign:                " << TimeSoftSign << std::endl;
    if (TimeSoftSignDerivative != 0.0) {
        std::cout << "SoftSignDerivative:      " << TimeSoftSignDerivative
                  << std::endl;
    }
    if (TimeGauss != 0.0)
        std::cout << "Gauss:                   " << TimeGauss << std::endl;
    if (TimeGaussDerivative != 0.0)
        std::cout << "GaussDerivative:         " << TimeGaussDerivative << std::endl;

    std::cout << std::endl;

    // Loss Functions.
    if (TimeMeanSquaredError != 0.0) {
        std::cout << "MeanSquaredError:              " << TimeMeanSquaredError
                  << std::endl;
    }
    if (TimeMeanSquaredErrorGradients != 0.0) {
        std::cout << "TimeMeanSquaredErrorGradients: "
                  << TimeMeanSquaredErrorGradients << std::endl;
    }
    if (TimeCrossEntropy != 0.0) {
        std::cout << "CrossEntropy:                  " << TimeCrossEntropy
                  << std::endl;
    }
    if (TimeCrossEntropyGradients != 0.0) {
        std::cout << "TimeCrossEntropyGradients:     " << TimeCrossEntropyGradients
                  << std::endl;
    }

    if (TimeSigmoidOutput != 0.0)
        std::cout << "TimeSigmoidOutput: " << TimeSigmoidOutput << std::endl;

    std::cout << std::endl;

    // Regularization
    if (TimeL1Regularization != 0.0)
        std::cout << "L1Regularization:                 " << TimeL1Regularization
                  << std::endl;
    if (TimeAddL1RegularizationGradients != 0.0)
        std::cout << "TimeAddL1RegularizationGradients: "
                  << TimeAddL1RegularizationGradients
                  << std::endl;
    if (TimeL2Regularization != 0.0)
        std::cout << "L2Regularization:                 " << TimeL2Regularization
                  << std::endl;
    if (TimeAddL2RegularizationGradients != 0.0)
        std::cout << "TimeAddL2RegularizationGradients: "
                  << TimeAddL2RegularizationGradients
                  << std::endl;
    std::cout << std::endl;

    // Dropout
    if (TimeDropout != 0.0)
        std::cout << "TimeDropout: " << TimeDropout << std::endl;
    std::cout << std::endl;

    // Synchronization
    if (TimeSynchronization != 0.0)
        std::cout << "TimeSynchronization: " << TimeSynchronization << std::endl;
    std::cout << std::endl;
}

} // namespace DNN
} // namespace TMVA

#endif
