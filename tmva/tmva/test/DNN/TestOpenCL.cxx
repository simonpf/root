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

   TMatrixT<OpenCLDouble_t> A(1000, 1000), B(2000, 2000);

   for (size_t i = 0; i < 1000; i++) {
      for (size_t j = 0; j < 1000; j++) {
         A(i,j) = 1.0;
      }
   }

   for (size_t i = 0; i < 2000; i++) {
      for (size_t j = 0; j < 2000; j++) {
         B(i,j) = 1.0;
      }
   }

   TOpenCLMatrix Ad(A), Bd(B);

   TOpenCL::Dropout(Ad, 0.5);
   TOpenCL::Dropout(Bd, 0.5);

   TMatrixT<OpenCLDouble_t> Ar(Ad);
   TMatrixT<OpenCLDouble_t> Br(Bd);

   std::cout << "A: " << Ar.Sum() / (1000.0 * 1000.0) << std::endl;
   std::cout << "B: " << Br.Sum() / (2000.0 * 2000.0) << std::endl;

}
