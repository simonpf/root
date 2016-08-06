// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh 04/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Declaration of the generic TDeviceDataLoader class which takes //
// care of the data transfer from the TMVA event data to the      //
// computing architecture.                                        //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_DEVICEDATALOADER
#define TMVA_DNN_DEVICEDATALOADER

#include <algorithm>
#include <iterator>
#include <utility>
#include "tbb/atomic.h"
#include "TMatrix.h"
#include "TMVA/Event.h"

namespace TMVA {
namespace DNN  {

// Input Data Types
using MatrixInput_t    = std::pair<const TMatrixT<Double_t> &,
                                   const TMatrixT<Double_t> &>;
using TMVAInput_t      = std::vector<Event*>;
using IndexIterator_t  = typename std::vector<size_t>::iterator;

/** TBatch class.
 *
 * Lightweight representation of a batch of data on the device. Holds
 * pointer to the input (samples) and output data (labels) as well as
 * to the data stream in which this batch is transferred.
 *
 * Provides GetInput() and GetOutput() member functions that return
 * TDeviceMatrix representations of the input and output data in the
 * batch.
 */
template<typename Architecture_t>
class TBatch
{
private:

   using DeviceBuffer_t = typename Architecture_t::Device_t::DeviceBuffer_t;
   using Matrix_t       = typename Architecture_t::Matrix_t;

   DeviceBuffer_t & fDeviceBuffer;  ///< Reference to the input data buffer.

   size_t fNInputFeatures;     ///< Number of input features.
   size_t fNOutputFeatures;    ///< Number of output features.
   size_t fBatchSize;          ///< Size of the batch.
   size_t fComputeStreamIndex; ///< The index of the corresponding compute stream.

public:

   TBatch(size_t batchSize,
          size_t nInputFeatures,
          size_t nOutputFeatures,
          size_t computeStreamIndex,
          DeviceBuffer_t & deviceBuffer)
    : fDeviceBuffer(deviceBuffer), fNInputFeatures(nInputFeatures),
    fNOutputFeatures(nOutputFeatures), fComputeStreamIndex(computeStreamIndex),
    fBatchSize(batchSize)
    {
       // Wait for previous computations to finish.
       fDeviceBuffer.SynchronizeComputation();
       fDeviceBuffer.SetUnconsumed();
    }
   TBatch(TBatch &&) = default;

   ~TBatch()
   {
      // Buffer goes out of scope.
      fDeviceBuffer.SetConsumed();
   }

    /** Return the batch input data as matrix corresponding to the architectures
     *  matrix type. The matrix is passed *  the data stream in which the async.
     *  data transfer to the corresponding buffer is performed, so that operations
     * on the matrix can synchronize with it. */
    Matrix_t GetInput();

    /** Return the outpur data as a Matrix_t. Also forwards the data stream in
     *  which the async. data transfer is performed to the matrix. See above.
     */
    Matrix_t GetOutput();
};

template<typename DataLoad_t>
class TBatchIterator;

/** TBatchIterator Class
 *
 * Class that implements an iterator over data sets. The
 * batch iterator has to take care of the preloading of the data which is
 * why a special implementation is required.
 */
template <typename DataLoader_t>
class TBatchIterator
{
private:

    using SampleIndexIterator_t = typename std::vector<size_t>::iterator;
    using Batch_t               = typename DataLoader_t::BatchType;

    DataLoader_t  & fDataLoader;  ///< Dataloader managing data transfer.
    size_t fBatchIndex;           ///< Index of this batch in the current epoch.

public:

    TBatchIterator(DataLoader_t &dataLoader,
                   size_t batchIndex = 0);

    /** Advance to the next batch and check if data should be preloaded. */
    TBatchIterator & operator++() {fBatchIndex++;}
    /** Return TDeviceBatch object corresponding to the current iterator position. */
    Batch_t operator*()           const {return fDataLoader.GetBatch();}

    bool operator!=(const TBatchIterator & other)
    {
       return fBatchIndex == other.fBatchIndex;
    }
};

/** The TDeviceDataLoader Class
 *
 * The TDeviceDataLoader class takes care of transferring training and test data
 * from the host to the device. The data transfer is performed asynchronously
 * and multiple data set batches can be transferred combined into transfer batches,
 * which contain a fixed number of data set batches. The range of the preloading
 * is defined in multiples of transfer batches. */
template <typename Architecture_t, typename Data_t>
class TDeviceDataLoader
{
private:

   using Device_t       = typename Architecture_t::Device_t;
   using HostBuffer_t   = typename Device_t::HostBuffer_t;
   using DeviceBuffer_t = typename Device_t::DeviceBuffer_t;
   using Matrix_t       = typename Architecture_t::Matrix_t;

   const Data_t & fInputData;

   size_t fNSamples;          ///< No. of samples in the data set.
   size_t fNInputFeatures;    ///< No. of features in input sample.
   size_t fNOutputFeatures;   ///< No. of features in output sample (truth).
   size_t fBatchSize;         ///< No. of samples in a (mini-)batch
   size_t fBufferSize;        ///< No. of elements in a batch-buffer.

   size_t fNBatchesInEpoch;   ///< No. of batches in one epoch.
   size_t fInputMatrixSize;   ///< No. of elements in input matrix.
   size_t fOutputMatrixSize;  ///< No. of elements in output matrix.

   Device_t fDevice;
   size_t   fNComputeStreams;
   size_t   fNDataStreams;

   tbb::atomic<size_t> fBatchIndex;

   std::vector<std::vector<HostBuffer_t>>   fHostBuffers;
   std::vector<std::vector<DeviceBuffer_t>> fDeviceBuffers;

   std::vector<size_t> fSampleIndices; ///< Shuffled sample indices.

public:

   using Batch_t         = TBatch<Architecture_t>;
   using BatchIterator_t = TBatchIterator<Architecture_t>;

   TDeviceDataLoader(const Data_t & inputData,
                     size_t nsamples,
                     size_t batchSize,
                     size_t ninputFeatures,
                     size_t noutputFeatures,
                     size_t fNDataStreams    = 5,
                     size_t fNComputeStreams = 1,
                     Device_t device         = Architecture_t::CreateDefaultDevice());

   ~TDeviceDataLoader() = default;

   /** Return iterator to batches in the training set. Samples in batches are
    *  are sampled randomly from the data set without replacement. */
   BatchIterator_t begin();
   BatchIterator_t end();

   Batch_t GetBatch();
   void    CopyBatch(const Data_t & data,
                     IndexIterator_t sampleIndexIterator,
                     size_t batchSize,
                     HostBuffer_t &buffer);

   size_t GetNBatchesInEpoch() const {return fNBatchesInEpoch;}
   size_t GetBatchSize()       const {return fBatchSize;}
   const Data_t & GetInputData() const {return fInputData;}

   void Shuffle() {std::random_shuffle(fSampleIndices.begin(), fSampleIndices.end());}
};

//
// Implementation
//____________________________________________________________________________
template <typename Architecture_t, typename Data_t>
TDeviceDataLoader<Architecture_t, Data_t>::TDeviceDataLoader(
    const Data_t & inputData,
    size_t nsamples,
    size_t batchSize,
    size_t nInputFeatures,
    size_t nOutputFeatures,
    size_t nDataStreams,
    size_t nComputeStreams,
    Device_t device)
    : fInputData(inputData), fNSamples(nsamples), fNInputFeatures(nInputFeatures),
      fNOutputFeatures(nOutputFeatures), fBatchSize(batchSize),
      fNDataStreams(nDataStreams), fNComputeStreams(nComputeStreams),
      fDevice(device), fBatchIndex(0)
{
   fNBatchesInEpoch = fNSamples / fBatchSize;
   fInputMatrixSize  = fNInputFeatures  * fBatchSize;
   fOutputMatrixSize = fNOutputFeatures * fBatchSize;
   fBufferSize       = fInputMatrixSize + fOutputMatrixSize;

   for (size_t i = 0; i < fNComputeStreams; i++) {
      fHostBuffers.push_back(std::vector<HostBuffer_t>());
      fDeviceBuffers.push_back(std::vector<DeviceBuffer_t>());
      for (size_t j = 0; j < fNDataStreams; j++) {
         fHostBuffers.back().push_back(fDevice.CreateHostBuffer(fBufferSize));
         fDeviceBuffers.back().push_back(fDevice.CreateDeviceBuffer(fBufferSize));
      }
   }

   fSampleIndices.reserve(fNSamples);
   for (size_t i = 0; i < fNSamples; i++) {
      fSampleIndices.push_back(i);
   }
}

/* template <typename Architecture_t, typename Data_t> */
/* TDeviceDataLoader<Architecture_t, Data_t>::CopyTransferBatch() */
/* { */
/*    size_t batchOffset = 0; */
/*    HostBuffer_t transferBuffer = fHostData[fStreamIndex]; */
/*    for (size_t i = 0; i < fTransferBatchSize; ++i) { */
/*       HostBuffer_t inputBuffer = transferBuffer.GetSubBuffer(batchOffset, */
/*                                                              fInputMatrixSize); */
/*       batchOffset += fInputMatrixSize; */
/*       HostBuffer_t outputBuffer = transferBuffer.GetSubBuffer(batchOffset, */
/*                                                               fOutputMatrixSize); */
/*       batchOffset += fOutputMatrixSize; */

/*       CopyBatch(data, fSampleIndices.begin() + fSampleIndexIndex, */
/*                 inputBuffer, outputBuffer); */
/*       fSampleIndexIndex += fBatchSize; */
/*    } */
/* } */

/* template <typename Architecture_t, typename Data_t> */
/* TDeviceDataLoader<Architecture_t, Data_t>::PrepareStream() */
/* { */
/*    for (fStreamIndex = 0; fStreamIndex < fPreloadOffset; fStreamIndex++) { */
/*       CopyTransferBatch(); */
/*       InvokeTransfer(); */
/*    } */
/* } */

/* template <typename Architecture_t, typename Data_t> */
/* TDeviceDataLoader<Architecture_t, Data_t>::InvokeDataTransfer() */
/* { */
/*    fStreamIndex = (fStreamIndex + 1) % (fPreloadOffset + 1); */
/*    size_t batchOffset = 0; */
/*    for (size_t i = 0; i < fTransferBatchSize; ++i) { */
/*       CopyTransferBatch(); */
/*    } */
/*    fDeviceData[fStreamIndex].Synchronize(); */
/*    fHostData[fStreamIndex].Transfer(fDeviceData[fStreamIndex]); */
/* } */

template <typename Architecture_t, typename Data_t>
auto TDeviceDataLoader<Architecture_t, Data_t>::GetBatch()
    -> Batch_t
{
   size_t taskIndex = (fBatchIndex++) % fNBatchesInEpoch;
   size_t computeStreamIndex = taskIndex % fNComputeStreams;
   size_t dataStreamIndex    = (taskIndex / fNComputeStreams) % fNDataStreams;

   HostBuffer_t & hostBuffer     = fHostBuffers[computeStreamIndex][dataStreamIndex];
   DeviceBuffer_t & deviceBuffer = fDeviceBuffers[computeStreamIndex][dataStreamIndex];
   hostBuffer.Lock();
   Batch_t batch(fBatchSize, fNInputFeatures, fNOutputFeatures,
                 computeStreamIndex, deviceBuffer);
   size_t sampleIndex = taskIndex * fBatchSize;
   CopyBatch(fInputData, fSampleIndices.begin() + sampleIndex,
             fBatchSize, hostBuffer);
   hostBuffer.CopyTo(deviceBuffer);
   hostBuffer.Release();

   return batch;
}

template <typename Architecture_t>
auto TBatch<Architecture_t>::GetInput()
    -> Matrix_t
{
   fDeviceBuffer.SynchronizeTransfer();
   size_t offset = 0;
   size_t size   = fBatchSize * fNInputFeatures;
   return Matrix_t(fBatchSize, fNInputFeatures,
                   fDeviceBuffer.GetSubBuffer(offset, size),
                   fComputeStreamIndex);
}

template <typename Architecture_t>
auto TBatch<Architecture_t>::GetOutput()
    -> Matrix_t
{
   fDeviceBuffer.SynchronizeTransfer();
   size_t offset = fBatchSize * fNInputFeatures;
   size_t size   = fBatchSize * fNOutputFeatures;
   return Matrix_t(fBatchSize, fNInputFeatures,
                   fDeviceBuffer.GetSubBuffer(offset, size),
                   fComputeStreamIndex);
}

} // namespace TMVA
} // namespace DNN

#endif
