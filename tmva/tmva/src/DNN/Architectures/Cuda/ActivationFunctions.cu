// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the activation functions for the TCuda      //
 // implementation of the low-level interface.                   //
 //////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "TMVA/DNN/Architectures/Cuda/Kernels.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::IdentityDerivative(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::IdentityDerivative<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeIdentityDerivative);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::Relu(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::Relu<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                                  (int) A.GetNrows(),
                                                  (int) A.GetNcols());
   tock(fTimings.TimeRelu);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::ReluDerivative(TCudaMatrix & B,
                                        const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::ReluDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeReluDerivative);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::Sigmoid(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::Sigmoid<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::SigmoidDerivative(TCudaMatrix & B,
                                           const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::SigmoidDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeSigmoidDerivative);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::Tanh(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();

   tick();
   ::TMVA::DNN::Cuda::Tanh<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeTanh);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::TanhDerivative(TCudaMatrix & B,
                                        const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();

   tick();
   ::TMVA::DNN::Cuda::TanhDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeTanhDerivative);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::SymmetricRelu(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::SymmetricRelu<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                                             (int) A.GetNrows(),
                                                             (int) A.GetNcols());
   tock(fTimings.TimeSymmetricRelu);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::SymmetricReluDerivative(TCudaMatrix & B,
                                                 const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::SymmetricReluDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeSymmetricReluDerivative);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::SoftSign(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::SoftSign<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeSoftSign);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::SoftSignDerivative(TCudaMatrix & B, const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();

   tick();
   ::TMVA::DNN::Cuda::SoftSignDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeSoftSignDerivative);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::Gauss(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();

   tick();
   ::TMVA::DNN::Cuda::Gauss<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeGauss);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::GaussDerivative(TCudaMatrix & B, const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   tick();
   ::TMVA::DNN::Cuda::GaussDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeGaussDerivative);
}

} // namespace DNN
} // namespace TMVA
