// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 27/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////
// Implementation of non-inline OpenCLMatrix class member functions. //
///////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL/OpenCLMatrix.h"

namespace TMVA {
namespace DNN  {

TOpenCLDevice TOpenCLMatrix::fDevice{};

TOpenCLMatrix::TOpenCLMatrix(size_t nRows, size_t nCols)
    : fNRows(nRows), fNCols(nCols), fNElements(nRows * nCols)
{
   cl_int err;
   fElementPointer = clCreateBuffer(fDevice.GetContext(), CL_MEM_READ_WRITE,
                                    fNElements * sizeof(OpenCLDouble_t),
                                    nullptr, &err);
   fDevice.HandleError(err);

}

TOpenCLMatrix::TOpenCLMatrix(const TMatrixT<OpenCLDouble_t> & A)
: fNRows(A.GetNrows()), fNCols(A.GetNcols()), fNElements(A.GetNoElements())
{
   OpenCLDouble_t * buffer = new OpenCLDouble_t[fNRows * fNCols];

   size_t bufferIndex = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         buffer[bufferIndex] = A(i,j);
         bufferIndex++;
      }
   }

   cl_int error;
   fElementPointer = clCreateBuffer(fDevice.GetContext(),
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    fNElements * sizeof(OpenCLDouble_t),
                                    (void*) buffer, &error);
   fDevice.HandleError(error);
}

TOpenCLMatrix::operator TMatrixT<OpenCLDouble_t>() const
{
   OpenCLDouble_t *buffer = new OpenCLDouble_t[fNRows * fNCols];

   cl_int err;
   err = clEnqueueReadBuffer(fDevice.GetQueue(), fElementPointer, CL_TRUE, 0,
                             fNElements * sizeof(OpenCLDouble_t), (void*) buffer, 0,
                             nullptr, nullptr);
   fDevice.HandleError(err);


   TMatrixT<OpenCLDouble_t> A(fNRows, fNCols);
   size_t bufferIndex = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         A(i,j) = buffer[bufferIndex];
         bufferIndex++;
      }
   }
   return A;
}

} // namespace DNN
} // namespace TMVA
