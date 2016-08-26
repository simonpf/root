// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
//  Concrete instantiation of the generic activation function test  //
//  for the OpenCL implementation of DNNs                           //
//////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/OpenCL.h"
#include "Utility.h"
#include "TestActivationFunctions.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing Activation Functions:" << std::endl;

    double error;

    // Identity.

    error = testIdentity<TOpenCL>(10);
    std::cout << "Testing identity activation:            ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testIdentityDerivative<TOpenCL>(10);
    std::cout << "Testing identity activation derivative: ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    // ReLU.

    error = testRelu<TOpenCL>(10);
    std::cout << "Testing ReLU activation:                ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testReluDerivative<TOpenCL>(10);
    std::cout << "Testing ReLU activation derivative:     ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    // Sigmoid.

    error = testSigmoid<TOpenCL>(10);
    std::cout << "Testing Sigmoid activation:             ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSigmoidDerivative<TOpenCL>(10);
    std::cout << "Testing Sigmoid activation derivative:  ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    // Tanh.

    error = testTanh<TOpenCL>(10);
    std::cout << "Testing Tanh activation:                ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testTanhDerivative<TOpenCL>(10);
    std::cout << "Testing Tanh activation derivative:     ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-7)
        return 1;

    // Symmetric Relu.

    error = testSymmetricRelu<TOpenCL>(10);
    std::cout << "Testing Sym. Relu activation:           ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSymmetricReluDerivative<TOpenCL>(10);
    std::cout << "Testing Sym. Relu activation derivative:";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    // Soft Sign.

    error = testSoftSign<TOpenCL>(10);
    std::cout << "Testing soft sign activation:           ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSoftSignDerivative<TOpenCL>(10);
    std::cout << "Testing soft sign activation derivative:";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    // Gauss.

    error = testGauss<TOpenCL>(10);
    std::cout << "Testing Gauss activation:               ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    error = testGaussDerivative<TOpenCL>(10);
    std::cout << "Testing Gauss activation derivative:    ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-10)
        return 1;

    return 0;
}
