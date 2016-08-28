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

#include "TMVA/DNN/DataLoader.h"
#include "TMVA/DNN/Architectures/OpenCL.h"
#include "TMVA/DNN/Architectures/OpenCL/OpenCLBuffers.h"
#include "TMVA/DNN/DataLoader.h"

namespace TMVA {
namespace DNN  {

//
// TOpenCLHostBuffer
//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCLHostBuffer<AFloat, AType>::TDestructor::operator()(AFloat **hostPointer)
{
   free(*hostPointer);
   delete[] hostPointer;
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLHostBuffer<AFloat, AType>::TOpenCLHostBuffer(size_t size)
    : fDevice(TOpenCL<AFloat, AType>::GetDefaultDevice()), fOffset(0)
{
    fComputeQueue = cl::CommandQueue(fDevice->GetContext(), fDevice->GetDevice());
    fBuffer = cl::Buffer(fDevice->GetContext(),
                         CL_MEM_READ_WRITE + CL_MEM_ALLOC_HOST_PTR,
                         size * sizeof(AFloat));
    fBufferPointer = (AFloat *) fComputeQueue.enqueueMapBuffer(
        fBuffer, CL_TRUE,
        CL_MAP_WRITE + CL_MAP_READ,
        0, size * sizeof(AFloat));
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLHostBuffer<AFloat, AType>::TOpenCLHostBuffer(
    size_t size,
    std::shared_ptr<TOpenCLDevice<AFloat, AType>> device)
    : fDevice(device), fOffset(0)
{
    fComputeQueue = cl::CommandQueue(fDevice->GetContext(), fDevice->GetDevice());
    fBuffer = cl::Buffer(fDevice->GetContext(),
                         CL_MEM_READ_WRITE + CL_MEM_ALLOC_HOST_PTR,
                         size * sizeof(AFloat));
    fBufferPointer = (AFloat *) fComputeQueue.enqueueMapBuffer(
        fBuffer, CL_TRUE,
        CL_MAP_WRITE + CL_MAP_READ,
        0, size * sizeof(AFloat));
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
auto TOpenCLHostBuffer<AFloat, AType>::GetSubBuffer(
    size_t offset,
    size_t size)
    -> TOpenCLHostBuffer
{
   TOpenCLHostBuffer buffer = *this;
   buffer.fOffset          += offset;
   return buffer;
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLHostBuffer<AFloat, AType>::operator AFloat * () const
{
   return fBufferPointer + fOffset;
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLHostBuffer<AFloat, AType>::operator cl::Buffer () const
{
   return fBuffer;
}

//
// TOpenCLDeviceBuffer
//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLDeviceBuffer<AFloat, AType>::TOpenCLDeviceBuffer(size_t size)
    : fSize(size), fOffset(0), fDevice(TOpenCL<AFloat, AType>::GetDefaultDevice())
{
   cl_int error;

   if (size > 0) {
   fBuffer = cl::Buffer(fDevice->GetContext(), CL_MEM_READ_WRITE,
                        size * sizeof(AFloat), nullptr, &error);
   }

   fDevice->HandleError(error);
   fComputeQueue = cl::CommandQueue(fDevice->GetContext(),
                                    fDevice->GetDevice(),
                                    0, &error);
   fDevice->HandleError(error);
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLDeviceBuffer<AFloat, AType>::TOpenCLDeviceBuffer(
    size_t size,
    std::shared_ptr<TOpenCLDevice<AFloat, AType>> device)
    : fSize(size), fOffset(0), fDevice(device)
{
   cl_int error;

   if (size > 0) {
   fBuffer = cl::Buffer(fDevice->GetContext(), CL_MEM_READ_WRITE,
                        size * sizeof(AFloat), nullptr, &error);
   }

   fDevice->HandleError(error);
   fComputeQueue = cl::CommandQueue(fDevice->GetContext(),
                                    fDevice->GetDevice(),
                                    0, &error);
   fDevice->HandleError(error);
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
TOpenCLDeviceBuffer<AFloat, AType>::operator cl::Buffer() const
{
   return fBuffer;
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
auto TOpenCLDeviceBuffer<AFloat, AType>::GetSubBuffer(
    size_t offset,
    size_t size)
    -> TOpenCLDeviceBuffer
{
   TOpenCLDeviceBuffer buffer;
   buffer.fSize         = size;
   buffer.fOffset       = fOffset;
   buffer.fDevice       = fDevice;
   buffer.fComputeQueue = fComputeQueue;

   _cl_buffer_region region;
   region.origin = offset * sizeof(AFloat);
   region.size   = size   * sizeof(AFloat);

   buffer.fBuffer = fBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                            CL_BUFFER_CREATE_TYPE_REGION,
                                            &region);
   return buffer;
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCLDeviceBuffer<AFloat, AType>::CopyFrom(
    const TOpenCLHostBuffer<AFloat, AType> &buffer
    ) const
{
   try {
       fComputeQueue = buffer.GetComputeQueue();
      void * temp = (void *) buffer.GetDataPointer();
//       fComputeQueue.enqueueCopyBuffer(buffer, fBuffer, 0, 0,
//                                       fSize * sizeof(AFloat));
      fComputeQueue.enqueueWriteBuffer(fBuffer, CL_FALSE, 0,
                                       fSize * sizeof(AFloat),
                                       buffer.GetDataPointer());
   fComputeQueue.flush();
   } catch(cl::Error error) {
      fDevice->HandleError(error.err());
   }
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCLDeviceBuffer<AFloat, AType>::CopyTo(
    const TOpenCLHostBuffer<AFloat, AType> &buffer
    ) const
{
   try {
      void * temp = (void *) buffer.GetDataPointer();
      fComputeQueue.enqueueReadBuffer(fBuffer, CL_TRUE, 0,
                                      fSize * sizeof(AFloat),
                                      buffer.GetDataPointer());
   } catch(cl::Error error) {
      fDevice->HandleError(error.err());
   }
   fComputeQueue.flush();
}

//______________________________________________________________________________
template<>
void TDataLoader<MatrixInput_t, TOpenCL<Double_t, EOpenCLDeviceType::kGpu>>::CopyInput(
    TOpenCLHostBuffer<Double_t, EOpenCLDeviceType::kGpu> & buffer,
    IndexIterator_t sampleIterator,
    size_t batchSize)
{
   const TMatrixT<Double_t> &inputMatrix  = std::get<0>(fData);
   size_t n = inputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = static_cast<Double_t>(inputMatrix(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template<>
void TDataLoader<MatrixInput_t, TOpenCL<Double_t, EOpenCLDeviceType::kGpu>>::CopyOutput(
    TOpenCLHostBuffer<Double_t, EOpenCLDeviceType::kGpu> & buffer,
    IndexIterator_t sampleIterator,
    size_t batchSize)
{
   const TMatrixT<Double_t> &outputMatrix  = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = static_cast<Double_t>(outputMatrix(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template<>
void TDataLoader<MatrixInput_t, TOpenCL<Real_t, EOpenCLDeviceType::kGpu>>::CopyInput(
    TOpenCLHostBuffer<Real_t, EOpenCLDeviceType::kGpu> & buffer,
    IndexIterator_t sampleIterator,
    size_t batchSize)
{
   const TMatrixT<Real_t> &inputMatrix  = std::get<0>(fData);
   size_t n = inputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = static_cast<Real_t>(inputMatrix(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template<>
void TDataLoader<MatrixInput_t, TOpenCL<Real_t, EOpenCLDeviceType::kGpu>>::CopyOutput(
    TOpenCLHostBuffer<Real_t, EOpenCLDeviceType::kGpu> & buffer,
    IndexIterator_t sampleIterator,
    size_t batchSize)
{
   const TMatrixT<Real_t> &outputMatrix  = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = static_cast<Real_t>(outputMatrix(sampleIndex, j));
      }
      sampleIterator++;
   }
}

template class TDataLoader<MatrixInput_t, TOpenCL<Real_t,  EOpenCLDeviceType::kGpu>>;
template class TDataLoader<MatrixInput_t, TOpenCL<Double_t, EOpenCLDeviceType::kGpu>>;

template class TOpenCLDeviceBuffer<Real_t,   EOpenCLDeviceType::kGpu>;
template class TOpenCLDeviceBuffer<Double_t, EOpenCLDeviceType::kGpu>;

template class TOpenCLHostBuffer<Real_t,   EOpenCLDeviceType::kGpu>;
template class TOpenCLHostBuffer<Double_t, EOpenCLDeviceType::kGpu>;

} // namespace TMVA
} // namespace DNN
