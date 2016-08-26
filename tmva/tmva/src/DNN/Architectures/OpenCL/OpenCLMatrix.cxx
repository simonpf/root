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

cl::Buffer TOpenCLMatrix::fRandomStreams{};
size_t     TOpenCLMatrix::fNStreams = 0;

//____________________________________________________________________________
TOpenCLMatrix::TOpenCLMatrix(size_t nRows,
                             size_t nCols)
    : fNRows(nRows), fNCols(nCols), fNElements(nRows * nCols),
      fElementBuffer(nRows * nCols)
{
   if (fNElements > 0) {
      InitializeRandomStreams();
   }
}

//____________________________________________________________________________
TOpenCLMatrix::TOpenCLMatrix(const TMatrixT<OpenCLDouble_t> & A)
    : fNRows(A.GetNrows()), fNCols(A.GetNcols()), fNElements(A.GetNoElements()),
      fElementBuffer(A.GetNoElements())
{
   if (fNElements > 0) {
      TOpenCLHostBuffer buffer(fNElements);
      size_t bufferIndex = 0;
      for (size_t j = 0; j < fNCols; j++) {
         for (size_t i = 0; i < fNRows; i++) {
            buffer[bufferIndex] = A(i,j);
            bufferIndex++;
         }
      }
      fElementBuffer.CopyFrom(buffer);
   }
   if (fNElements > 0) {
      InitializeRandomStreams();
   }
}

//____________________________________________________________________________
TOpenCLMatrix::TOpenCLMatrix(size_t nRows,
                             size_t nCols,
                             const TOpenCLDeviceBuffer & buffer)
    : fNRows(nRows), fNCols(nCols), fNElements(nRows * nCols),
      fElementBuffer(buffer)
{
   // Nothing to do here.
}

//____________________________________________________________________________
TOpenCLMatrix::operator TMatrixT<OpenCLDouble_t>() const
{
    std::cout << "nelements: " << fNElements << std::endl;
   TOpenCLHostBuffer buffer(fNElements);
   fElementBuffer.CopyTo(buffer);

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

//____________________________________________________________________________
TOpenCLMatrix & TOpenCLMatrix::operator=(const TMatrixT<OpenCLDouble_t> &A)
{
   fNRows = A.GetNrows();
   fNCols = A.GetNcols();
   fNElements = fNRows * fNCols;

   TOpenCLHostBuffer buffer(fNElements);

   size_t bufferIndex = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         buffer[bufferIndex] = A(i,j);
         bufferIndex++;
      }
   }

   fElementBuffer = TOpenCLDeviceBuffer(fNElements);
   fElementBuffer.CopyFrom(buffer);

   if (fNElements > 0) {
      InitializeRandomStreams();
   }
}

//____________________________________________________________________________
inline void TOpenCLMatrix::InitializeRandomStreams()
{
   clrngLfsr113Stream * streamBuffer = nullptr;
   size_t streamBufferSize;

   if ((fNStreams == 0) || (fNStreams < fNElements)) {
      std::cout << "Allocated " << fNElements << " random streams." << std::endl;

      // Create random streams.
      streamBuffer = clrngLfsr113CreateStreams(NULL, fNElements,
                                               &streamBufferSize, nullptr);

      // Transfer to device.
      try {
         fRandomStreams = cl::Buffer(fElementBuffer.GetDevice().GetContext(),
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     streamBufferSize, streamBuffer);
      } catch (cl::Error error) {
         fElementBuffer.GetDevice().HandleError(error.err());
      }

      fNStreams = fNElements;
      // Clean up.
      delete[] streamBuffer;
   }
}

} // namespace DNN
} // namespace TMVA
