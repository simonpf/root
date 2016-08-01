// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Test for the loss function implementation of the OpenCL backend //
// using the generic tests from TestLossFunctions.h                //
/////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TestLossFunctions.h"
#include "TMVA/DNN/Architectures/OpenCL.h"

using namespace TMVA::DNN;

int main()
{
   double error;

   //
   // Mean Squared Error.
   //

   error = testMeanSquaredError<TOpenCL>(10);
   std::cout << "Testing mean squared error loss:     ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

   error = testMeanSquaredErrorGradients<TOpenCL>(10);
   std::cout << "Testing mean squared error gradient: ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

   error = testCrossEntropy<TOpenCL>(10);
   std::cout << "Testing cross entropy loss:          ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

    error = testCrossEntropyGradients<TOpenCL>(10);
    std::cout << "Testing cross entropy gradient:      ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

   return 0;
}
