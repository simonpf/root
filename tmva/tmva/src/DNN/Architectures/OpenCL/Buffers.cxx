// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh 11/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Host and device buffer classes for the OpenCL implementation of //
// deep neural networks.                                           //
/////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"
#include "TMVA/DNN/Architectures/OpenCL/Buffers.h"
#include "TMVA/DNN/DataLoader.h"

namespace TMVA {
namespace DNN  {

//
// TOpenCLHostBuffer
//______________________________________________________________________________
void TOpenCLHostBuffer::TDestructor::operator()(OpenCLDouble_t **hostPointer)
{
   free(*hostPointer);
   delete[] hostPointer;
}

//______________________________________________________________________________
TOpenCLHostBuffer::TOpenCLHostBuffer(size_t size)
    : fDevice(TOpenCL::GetDefaultDevice()), fOffset(0)
{
    fComputeQueue = cl::CommandQueue(fDevice->GetContext(), fDevice->GetDevice());
    fBuffer = cl::Buffer(fDevice->GetContext(),
                         CL_MEM_READ_WRITE + CL_MEM_ALLOC_HOST_PTR,
                         size * sizeof(OpenCLDouble_t));
    fBufferPointer = (OpenCLDouble_t *) fComputeQueue.enqueueMapBuffer(
        fBuffer, CL_TRUE,
        CL_MAP_WRITE + CL_MAP_READ,
        0, size * sizeof(OpenCLDouble_t));
}

//______________________________________________________________________________
TOpenCLHostBuffer::TOpenCLHostBuffer(size_t size,
                                     std::shared_ptr<TOpenCLDevice> device)
    : fDevice(device), fOffset(0)
{
    fComputeQueue = cl::CommandQueue(fDevice->GetContext(), fDevice->GetDevice());
    fBuffer = cl::Buffer(fDevice->GetContext(),
                         CL_MEM_READ_WRITE + CL_MEM_ALLOC_HOST_PTR,
                         size * sizeof(OpenCLDouble_t));
    fBufferPointer = (OpenCLDouble_t *) fComputeQueue.enqueueMapBuffer(
        fBuffer, CL_TRUE,
        CL_MAP_WRITE + CL_MAP_READ,
        0, size * sizeof(OpenCLDouble_t));
}

//______________________________________________________________________________
TOpenCLHostBuffer TOpenCLHostBuffer::GetSubBuffer(size_t offset, size_t size)
{
   TOpenCLHostBuffer buffer = *this;
   buffer.fOffset          += offset;
   return buffer;
}

//______________________________________________________________________________
TOpenCLHostBuffer::operator OpenCLDouble_t * () const
{
   return fBufferPointer + fOffset;
}

//______________________________________________________________________________
TOpenCLHostBuffer::operator cl::Buffer () const
{
   return fBuffer;
}

//
// TOpenCLDeviceBuffer
//______________________________________________________________________________
TOpenCLDeviceBuffer::TOpenCLDeviceBuffer(size_t size)
    : fSize(size), fOffset(0), fDevice(TOpenCL::GetDefaultDevice())
{
   cl_int error;

   if (size > 0) {
   fBuffer = cl::Buffer(fDevice->GetContext(), CL_MEM_READ_WRITE,
                        size * sizeof(OpenCLDouble_t), nullptr, &error);
   }

   fDevice->HandleError(error);
   fComputeQueue = cl::CommandQueue(fDevice->GetContext(),
                                    fDevice->GetDevice(),
                                    0, &error);
   fDevice->HandleError(error);
}

//______________________________________________________________________________
TOpenCLDeviceBuffer::TOpenCLDeviceBuffer(size_t size,
                                         std::shared_ptr<TOpenCLDevice> device)
    : fSize(size), fOffset(0), fDevice(device)
{
   cl_int error;

   if (size > 0) {
   fBuffer = cl::Buffer(fDevice->GetContext(), CL_MEM_READ_WRITE,
                        size * sizeof(OpenCLDouble_t), nullptr, &error);
   }

   fDevice->HandleError(error);
   fComputeQueue = cl::CommandQueue(fDevice->GetContext(),
                                    fDevice->GetDevice(),
                                    0, &error);
   fDevice->HandleError(error);
}

//______________________________________________________________________________
TOpenCLDeviceBuffer::operator cl::Buffer() const
{
   return fBuffer;
}

//______________________________________________________________________________
TOpenCLDeviceBuffer TOpenCLDeviceBuffer::GetSubBuffer(size_t offset, size_t size)
{
    TOpenCLDeviceBuffer buffer;
   buffer.fSize         = size;
   buffer.fOffset       = fOffset;
   buffer.fDevice       = fDevice;
   buffer.fComputeQueue = fComputeQueue;

   _cl_buffer_region region;
   region.origin = offset * sizeof(OpenCLDouble_t);
   region.size   = size   * sizeof(OpenCLDouble_t);

   buffer.fBuffer = fBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                            CL_BUFFER_CREATE_TYPE_REGION,
                                            &region);
   return buffer;
}

//______________________________________________________________________________
void TOpenCLDeviceBuffer::CopyFrom(const TOpenCLHostBuffer &buffer) const
{
   try {
       fComputeQueue = buffer.GetComputeQueue();
      void * temp = (void *) buffer.GetDataPointer();
//       fComputeQueue.enqueueCopyBuffer(buffer, fBuffer, 0, 0,
//                                       fSize * sizeof(OpenCLDouble_t));
      fComputeQueue.enqueueWriteBuffer(fBuffer, CL_FALSE, 0,
                                       fSize * sizeof(OpenCLDouble_t),
                                       buffer.GetDataPointer());
   fComputeQueue.flush();
   } catch(cl::Error error) {
      fDevice->HandleError(error.err());
   }
}

//______________________________________________________________________________
void TOpenCLDeviceBuffer::CopyTo(const TOpenCLHostBuffer &buffer) const
{
   try {
      void * temp = (void *) buffer.GetDataPointer();
      fComputeQueue.enqueueReadBuffer(fBuffer, CL_TRUE, 0,
                                      fSize * sizeof(OpenCLDouble_t),
                                      buffer.GetDataPointer());
   } catch(cl::Error error) {
      fDevice->HandleError(error.err());
   }
   fComputeQueue.flush();
}

//______________________________________________________________________________
template<>
void TDataLoader<MatrixInput_t, TOpenCL>::CopyInput(
    TOpenCLHostBuffer & buffer,
    IndexIterator_t sampleIterator,
    size_t batchSize)
{
   const TMatrixT<Double_t> &inputMatrix  = std::get<0>(fData);
   size_t n = inputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = inputMatrix(sampleIndex, j);
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template<>
void TDataLoader<MatrixInput_t, TOpenCL>::CopyOutput(
    TOpenCLHostBuffer & buffer,
    IndexIterator_t sampleIterator,
    size_t batchSize)
{
   const TMatrixT<Double_t> &outputMatrix  = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = outputMatrix(sampleIndex, j);
      }
      sampleIterator++;
   }
}

template class TDataLoader<MatrixInput_t, TOpenCL>;

} // namespace TMVA
} // namespace DNN
