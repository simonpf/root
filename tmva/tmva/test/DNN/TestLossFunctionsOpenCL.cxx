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
   using Scalar_t = Real_t;
   using OpenCL_t = TOpenCL<Scalar_t, EOpenCLDeviceType::kGpu>;

   std::cout << "Testing loss functions:" << std::endl;

   //
   // Mean Squared Error.
   //

   Double_t error = testMeanSquaredError<OpenCL_t>(10);
   std::cout << "Testing mean squared error loss:     ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3)
       return 1;

   error = testMeanSquaredErrorGradients<OpenCL_t>(10);
   std::cout << "Testing mean squared error gradient: ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3)
       return 1;

   error = testCrossEntropy<OpenCL_t>(10);
   std::cout << "Testing cross entropy loss:          ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3)
       return 1;

    error = testCrossEntropyGradients<OpenCL_t>(10);
    std::cout << "Testing cross entropy gradient:      ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-3)
        return 1;

   return 0;
}
