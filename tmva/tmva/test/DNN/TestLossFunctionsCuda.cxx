// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Test for the loss function reference implementation using the //
// generic test defined in TestLossFunctions.h.                  //
///////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestLossFunctions.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing Loss Functions:" << std::endl << std::endl;

    double error;

    //
    // Mean Squared Error.
    //

    error = testMeanSquaredError<TCuda<false>>(10);
    std::cout << "Testing mean squared error loss:     ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testMeanSquaredErrorGradients<TCuda<false>>(10);
    std::cout << "Testing mean squared error gradient: ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    //
    // Cross Entropy.
    //

    error = testCrossEntropy<TCuda<false>>(10);
    std::cout << "Testing cross entropy loss:          ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testCrossEntropyGradients<TCuda<false>>(10);
    std::cout << "Testing mean squared error gradient: ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;
}