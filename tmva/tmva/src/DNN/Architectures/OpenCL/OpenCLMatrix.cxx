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

cl::Buffer    TOpenCLMatrix::fRandomStreams{};
size_t        TOpenCLMatrix::fNStreams = 0;

TOpenCLMatrix::TOpenCLMatrix(size_t nRows,
                             size_t nCols,
                             const TOpenCLDevice & device,
                             size_t computeStreamIndex)
    : fDevice(device), fNRows(nRows), fNCols(nCols), fNElements(nRows * nCols),
      fComputeStreamIndex(computeStreamIndex)
{
   if (fNElements > 0) {
      try {
         fElementBuffer = cl::Buffer(fDevice.GetContext(), CL_MEM_READ_WRITE,
                                     fNElements * sizeof(OpenCLDouble_t));
      } catch (cl::Error error) {
         std::cout << "Error allocating TOpenCLMatrix: 1 " << error.what() << std::endl;
         std::cout << nRows << " x " << nCols << std::endl;
         fDevice.HandleError(error.err());
      }
   }
   InitializeRandomStreams();
}

TOpenCLMatrix::TOpenCLMatrix(const TMatrixT<OpenCLDouble_t> & A,
                             const TOpenCLDevice & device,
                             size_t computeStreamIndex)
    : fDevice(device), fNRows(A.GetNrows()), fNCols(A.GetNcols()),
      fNElements(A.GetNoElements()), fComputeStreamIndex(computeStreamIndex)
{
   if (fNElements > 0) {
      OpenCLDouble_t * buffer = new OpenCLDouble_t[fNRows * fNCols];
      size_t bufferIndex = 0;
      for (size_t j = 0; j < fNCols; j++) {
         for (size_t i = 0; i < fNRows; i++) {
            buffer[bufferIndex] = A(i,j);
            bufferIndex++;
         }
      }

      try {
         fElementBuffer = cl::Buffer(fDevice.GetContext(),
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     fNElements * sizeof(OpenCLDouble_t), buffer);
      } catch (cl::Error error) {
         std::cout << "Error allocating TOpenCLMatrix: 2 " << error.what()
                   << std::endl << std::endl;
         fDevice.HandleError(error.err());
      }

      delete[] buffer;
   }
   InitializeRandomStreams();
}

TOpenCLMatrix::TOpenCLMatrix(size_t nRows,
                             size_t nCols,
                             const DeviceBuffer_t & buffer,
                             size_t computeStreamIndex)
    : fDevice(buffer.GetDevice()), fNRows(nRows), fNCols(nCols),
      fNElements(nRows * nCols), fElementBuffer(buffer.GetBuffer()),
      fComputeStreamIndex(computeStreamIndex)
{
   // Nothing to do here.
}

TOpenCLMatrix::operator TMatrixT<OpenCLDouble_t>() const
{
   OpenCLDouble_t *buffer = new OpenCLDouble_t[fNRows * fNCols];

   std::cout << "element buffer " << fElementBuffer() << std::endl;
   try{
   fDevice.GetQueue(0).enqueueReadBuffer(fElementBuffer, CL_TRUE,
                                         0, fNElements * sizeof(OpenCLDouble_t),
                                         (void *) buffer);
   } catch (cl::Error error) {
       fDevice.HandleError(error.err());
   }

   TMatrixT<OpenCLDouble_t> A(fNRows, fNCols);
   size_t bufferIndex = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         A(i,j) = buffer[bufferIndex];
         bufferIndex++;
      }
   }

   delete[] buffer;
   return A;
}

TOpenCLMatrix & TOpenCLMatrix::operator=(const TMatrixT<OpenCLDouble_t> &A)
{
   fNRows = A.GetNrows();
   fNCols = A.GetNcols();
   fNElements = fNRows * fNCols;

   OpenCLDouble_t *buffer = new OpenCLDouble_t[fNRows * fNCols];

   size_t bufferIndex = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         buffer[bufferIndex] = A(i,j);
         bufferIndex++;
      }
   }

   try {
      fElementBuffer = cl::Buffer(fDevice.GetContext(),
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  fNElements * sizeof(OpenCLDouble_t), buffer);
   } catch (cl::Error error) {
      std::cout << "Error allocating TOpenCLMatrix: = " << error.what()
                << std::endl << std::endl;
      fDevice.HandleError(error.err());
   }

   delete[] buffer;
   InitializeRandomStreams();
}

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
      fRandomStreams = cl::Buffer(fDevice.GetContext(),
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  streamBufferSize, streamBuffer);
      } catch (cl::Error error) {
         fDevice.HandleError(error.err());
      }

      fNStreams = fNElements;
      // Clean up.
      delete[] streamBuffer;
   }
}

} // namespace DNN
} // namespace TMVA
