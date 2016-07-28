// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 26/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////
// Test for general OpenCL functionality. //
////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"
#include "Utility.h"
#include "TMatrix.h"

using namespace TMVA::DNN;

int main()
{

   TMatrixT<OpenCLDouble_t> A(10, 10), B(10, 10), C(10, 10);
   randomMatrix(A);
   randomMatrix(B);

   TOpenCLMatrix Ad(A), Bd(B), Cd(10, 10);

   C.Mult(A, B);
   TOpenCL::Multiply(Cd, Ad, Bd);

   TMatrixT<OpenCLDouble_t> Cr(Cd);

   C.Print();
   Cr.Print();

   std::cout << "Multiplication error:" << maximumRelativeError(Cr, C)
             << std::endl;

}
