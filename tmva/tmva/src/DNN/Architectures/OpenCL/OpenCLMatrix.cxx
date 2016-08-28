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

template<typename AFloat, EOpenCLDeviceType AType>
cl::Buffer TOpenCLMatrix<AFloat, AType>::fRandomStreams{};
template<typename AFloat, EOpenCLDeviceType AType>
size_t     TOpenCLMatrix<AFloat, AType>::fNStreams = 0;

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLMatrix<AFloat, AType>::TOpenCLMatrix(size_t nRows,
                             size_t nCols)
    : fNRows(nRows), fNCols(nCols), fNElements(nRows * nCols),
      fElementBuffer(nRows * nCols)
{
   if (fNElements > 0) {
      InitializeRandomStreams();
   }
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLMatrix<AFloat, AType>::TOpenCLMatrix(const TMatrixT<Double_t> & A)
    : fNRows(A.GetNrows()), fNCols(A.GetNcols()), fNElements(A.GetNoElements()),
      fElementBuffer(A.GetNoElements())
{
   if (fNElements > 0) {
      TOpenCLHostBuffer<AFloat, AType> buffer(fNElements);
      size_t bufferIndex = 0;
      for (size_t j = 0; j < fNCols; j++) {
         for (size_t i = 0; i < fNRows; i++) {
            buffer[bufferIndex] = static_cast<AFloat>(A(i,j));
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
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLMatrix<AFloat, AType>::TOpenCLMatrix(
    const TOpenCLDeviceBuffer<AFloat, AType> & buffer,
    size_t nRows,
    size_t nCols)
    : fNRows(nRows), fNCols(nCols), fNElements(nRows * nCols),
      fElementBuffer(buffer)
{
   // Nothing to do here.
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLMatrix<AFloat, AType>::operator TMatrixT<Double_t>() const
{
   TOpenCLHostBuffer<AFloat, AType> buffer(fNElements);
   fElementBuffer.CopyTo(buffer);

   TMatrixT<Double_t> A(fNRows, fNCols);
   size_t bufferIndex = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         A(i,j) = static_cast<Double_t>(buffer[bufferIndex]);
         bufferIndex++;
      }
   }
   return A;
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
auto TOpenCLMatrix<AFloat, AType>::operator=(const TMatrixT<Double_t> &A)
    -> TOpenCLMatrix &
{
   fNRows = A.GetNrows();
   fNCols = A.GetNcols();
   fNElements = fNRows * fNCols;

   TOpenCLHostBuffer<AFloat, AType> buffer(fNElements);

   size_t bufferIndex = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         buffer[bufferIndex] = A(i,j);
         bufferIndex++;
      }
   }

   fElementBuffer = TOpenCLDeviceBuffer<AFloat, AType>(fNElements);
   fElementBuffer.CopyFrom(buffer);

   if (fNElements > 0) {
      InitializeRandomStreams();
   }
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
inline void TOpenCLMatrix<AFloat, AType>::InitializeRandomStreams()
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

template class TOpenCLMatrix<Real_t,   EOpenCLDeviceType::kGpu>;
template class TOpenCLMatrix<Double_t, EOpenCLDeviceType::kGpu>;

} // namespace DNN
} // namespace TMVA
