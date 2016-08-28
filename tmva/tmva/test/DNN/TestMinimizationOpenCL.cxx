// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 07/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Test the OpenCL implementation of neural network by learning a //
// randomly generated linear mapping from a 20-dimensional input  //
// space to a 1-dimensional output space represented by a         //
// vector. This vector should be reproduced as the output of      //
// network to an identity matrix input.                           //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/OpenCL.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

template<typename AFloat>
using OpenCL_t = TOpenCL<AFloat, EOpenCLDeviceType::kGpu>;

int main ()
{
   std::cout << "Testing minimization: (single precision)" << std::endl;

   Double_t error = testMinimization<OpenCL_t<Real_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
       return 1;
   }

   error = testMinimizationMomentum<OpenCL_t<Real_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
       return 1;
   }
   std::cout << std::endl << "Testing minimization: (double precision)" << std::endl;

   error = testMinimization<OpenCL_t<Double_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > 1e-5) {
       return 1;
   }

   error = testMinimizationMomentum<OpenCL_t<Double_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
   if (error > 1e-5) {
       return 1;
   }
   return 0;
}

