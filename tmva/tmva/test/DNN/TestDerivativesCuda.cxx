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
// Concrete instantiation of the generic derivative test for the //
//  reference implementation.                                    //
///////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestDerivatives.h"

using namespace TMVA::DNN;

int main()
{
    double error;

    //
    // Activation Functions
    //

    std::cout << "Activation Functions:" << std::endl;
    error = testActivationFunctionDerivatives<TCuda>();
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << print_error(error);
    std::cout << std::endl << std::endl;
    if (error > 1e-5)
        return 1;

    //
    // Loss Functions
    //

    std::cout << "Loss Functions:" << std::endl;
    error = testLossFunctionGradients<TCuda>();
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << print_error(error);
    std::cout << std::endl << std::endl;
    if (error > 1e-5)
        return 1;

    //
    // Regularization Functions
    //

    std::cout << "Regularization:" << std::endl;
    error = testRegularizationGradients<TCuda>();
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << print_error(error);
    std::cout << std::endl << std::endl;
    if (error > 1e-5)
        return 1;

    return 0;
}
