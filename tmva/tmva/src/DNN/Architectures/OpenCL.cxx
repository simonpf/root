// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 05/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Implementation of the specialization of the CopyBatch member //
// functions of the TDeviceDataLoader class for OpenCL          //
// architectures.                                               //
//////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"

namespace TMVA {
namespace DNN  {

template <>
void TDeviceDataLoader<TOpenCL, MatrixInput_t>::CopyBatch(
    const MatrixInput_t & data,
    IndexIterator_t sampleIndexIteratorBegin,
    size_t batchSize,
    TOpenCLDevice::HostBuffer_t & buffer)
{
   const TMatrixT<Double_t> &inputData  = std::get<0>(data);
   const TMatrixT<Double_t> &outputData = std::get<1>(data);

   size_t nInput  = inputData.GetNcols();
   size_t nOutput = outputData.GetNcols();

   // Copy input matrix;

   IndexIterator_t sampleIndexIterator = sampleIndexIteratorBegin;
   for (size_t i = 0; i < batchSize; i++) {
       size_t sampleIndex = *sampleIndexIterator;
       // Copy input matrices.
       for (size_t j = 0; j < nInput; j++) {
           size_t bufferIndex = j * batchSize + i;
           buffer[bufferIndex] = inputData(sampleIndex, j);
       }
       sampleIndexIterator++;
   }

   // Copy output matrix;

   size_t offset = nInput * batchSize;
   sampleIndexIterator = sampleIndexIteratorBegin;
   for (size_t i = 0; i < batchSize; i++) {
       size_t sampleIndex = *sampleIndexIterator;
       // Copy input matrices
       for (size_t j = 0; j < nOutput; j++) {
           size_t bufferIndex = j * batchSize + i;
           buffer[offset + bufferIndex] = outputData(sampleIndex, j);
       }
       sampleIndexIterator++;
   }
}

template
void TDeviceDataLoader<TOpenCL, MatrixInput_t>::CopyBatch(
    const MatrixInput_t & data,
    IndexIterator_t sampleIndexIteratorBegin,
    size_t batchSize,
    TOpenCLDevice::HostBuffer_t & buffer);

} // namespace TMVA
} // namespace DNN
