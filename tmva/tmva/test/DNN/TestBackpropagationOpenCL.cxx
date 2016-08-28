// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Concrete instantiation of the generic backpropagation test for //
// OpenCL architectures.                                          //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/OpenCL.h"
#include "TMatrix.h"
#include "TestBackpropagation.h"

using namespace TMVA::DNN;

int main()
{
   using Scalar_t = Real_t;
   using OpenCL_t = TOpenCL<Scalar_t, EOpenCLDeviceType::kGpu>;

   std::cout << "Testing Backpropagation:" << std::endl;

   double error;

   error = testBackpropagationWeightsLinear<OpenCL_t>(1.0);
   if (error > 1)
       return 1;

   error = testBackpropagationL1Regularization<OpenCL_t>(1e-2);
   if (error > 1)
       return 1;

   error = testBackpropagationL2Regularization<OpenCL_t>(1.0);
   if (error > 1)
       return 1;

   error = testBackpropagationBiasesLinear<OpenCL_t>(1.0);
   if (error > 1)
       return 1;

   return 0;
}
