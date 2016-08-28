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
// Concrete instantiation of the generic matrix arithmetic test for //
// OpenCL architectures.                                            //
//////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/OpenCL.h"
#include "TestMatrixArithmetic.h"

using namespace TMVA::DNN;

template<typename AFloat>
using OpenCL_t = TOpenCL<AFloat, EOpenCLDeviceType::kGpu>;

int main()
{

   std::cout << "Testing CPU matrix arithmetic (double):" << std::endl;

   Double_t error = testMultiplication<OpenCL_t<Double_t>>(10);
   std::cout << "Multiplication: " << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
       return 1;

   error = testSumColumns<OpenCL_t<Double_t>>(1);
   std::cout << "Column Sum:     " << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
       return 1;

   std::cout << "Testing CPU matrix arithmetic (float):" << std::endl;

   error = testMultiplication<OpenCL_t<Real_t>>(10);
   std::cout << "Multiplication: " << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
       return 1;

   error = testSumColumns<OpenCL_t<Real_t>>(1);
   std::cout << "Column Sum:     " << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
       return 1;
}
