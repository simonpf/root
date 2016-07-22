// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
//  Concrete instantiation of the generic activation function test  //
//  for the multi-threaded CPU implementation.                      //
//////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "Utility.h"
#include "TestActivationFunctions.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing Activation Functions:" << std::endl;

    double error;

    // Identity.

    error = testIdentity<TCpu<Double_t,false>>(10);
    std::cout << "Testing identity activation:            ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testIdentityDerivative<TCpu<Double_t,false>>(10);
    std::cout << "Testing identity activation derivative: ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    // ReLU.

    error = testRelu<TCpu<Double_t,false>>(10);
    std::cout << "Testing ReLU activation:                ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testReluDerivative<TCpu<Double_t,false>>(10);
    std::cout << "Testing ReLU activation derivative:     ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    // Sigmoid.

    error = testSigmoid<TCpu<Double_t,false>>(10);
    std::cout << "Testing Sigmoid activation:             ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSigmoidDerivative<TCpu<Double_t,false>>(10);
    std::cout << "Testing Sigmoid activation derivative:  ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    // TanH.

    error = testTanh<TCpu<Double_t, false>>(10);
    std::cout << "Testing TanH activation:                   ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testTanhDerivative<TCpu<Double_t, false>>(10);
    std::cout << "Testing TanH activation derivative:        ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Symmetric ReLU.

    error = testSymmetricRelu<TCpu<Double_t, false>>(10);
    std::cout << "Testing Symm. ReLU activation:             ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSymmetricReluDerivative<TCpu<Double_t, false>>(10);
    std::cout << "Testing Symm. ReLU activation derivative:  ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Soft Sign.

    error = testSoftSign<TCpu<Double_t, false>>(10);
    std::cout << "Testing Soft Sign activation:              ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSoftSignDerivative<TCpu<Double_t, false>>(10);
    std::cout << "Testing Soft Sign activation derivative:   ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Gauss.

    error = testGauss<TCpu<Double_t, false>>(10);
    std::cout << "Testing Gauss activation:                  ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testGaussDerivative<TCpu<Double_t, false>>(10);
    std::cout << "Testing Gauss activation derivative:       ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    return 0;
}