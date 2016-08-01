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

int main()
{
   std::cout << "Testing OpenCL Matrix Arithmetic:" << std::endl;

   Double_t error = testMultiplication<TOpenCL>(10);
   std::cout << "Multiplication: " << "Max. rel. error: " << error << std::endl;

   error = testSumColumns<TOpenCL>(10);
   std::cout << "Column Sum:     " << "Max. rel. error: " << error << std::endl;
}
