// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 02/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 ///////////////////////////////////////////////////////////////
 // Implementation of the initialization functions for OpenCL //
 // Architectures                                             //
 ///////////////////////////////////////////////////////////////

#include "TRandom.h"
#include "TMatrixT.h"
#include "TMVA/DNN/Architectures/OpenCL.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
void TOpenCL::InitializeGauss(TOpenCLMatrix & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TMatrixT<OpenCLDouble_t> B(m, n);
   TRandom rand(time(nullptr));
   Real_t sigma = sqrt(2.0 / ((Real_t) n));

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         B(i,j) = rand.Gaus(0.0, sigma);
      }
   }

   A = B;
}

//______________________________________________________________________________
void TOpenCL::InitializeUniform(TOpenCLMatrix & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TMatrixT<OpenCLDouble_t> B(m, n);
   TRandom rand(time(nullptr));
   Real_t range = sqrt(2.0 / ((Real_t) n));

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         B(i,j) = rand.Uniform(-range, range);
      }
   }
   A = B;
}

//______________________________________________________________________________
void TOpenCL::InitializeIdentity(TOpenCLMatrix & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();
   TMatrixT<OpenCLDouble_t> B(m, n);

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n ; j++) {
         B(i,j) = 0.0;
      }

      if (i < n) {
         B(i,i) = 1.0;
      }
   }
   A = B;
}

//______________________________________________________________________________
void TOpenCL::InitializeZero(TOpenCLMatrix & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();
   TMatrixT<OpenCLDouble_t> B(m, n);

   for (size_t i = 0; i < m * n; i++) {
      for (size_t j = 0; j < n ; j++) {
         B(i,j) = 0.0;
      }
   }
   A = B;
}

} // namespace DNN
} // namespace TMVA
