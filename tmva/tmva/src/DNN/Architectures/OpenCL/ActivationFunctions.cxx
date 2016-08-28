// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 01/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Activation functions for OpenCL implementation of deep neural //
// networks.                                                     //
///////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"

namespace TMVA {
namespace DNN  {

/** Launch an OpenCL kernel that maps the given Kernel to the matrix A, by
 *  launching one workgroup per column. */
//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
inline void ExecuteActivationFunctionKernel(
    EOpenCLKernel kernel,
    TOpenCLMatrix<AFloat, AType> & A)
{
   const TOpenCLDevice<AFloat, AType> & device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);

   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(kernel, queue, global, local, A.GetElementBuffer(), m);
}

/** Launch an OpenCL kernel that applies an activation function derivative kernel
 *  to the two matrices by launching one workgroup per column */
//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
inline void ExecuteActivationFunctionDerivativeKernel(
          EOpenCLKernel kernel,
          TOpenCLMatrix<AFloat, AType> & B,
    const TOpenCLMatrix<AFloat, AType> & A)
{
   const TOpenCLDevice<AFloat, AType> & device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);
   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(kernel, queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(), m);
   B.SetComputeQueue(queue);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Identity(TOpenCLMatrix<AFloat, AType> & /*A*/) {}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::IdentityDerivative(
          TOpenCLMatrix<AFloat, AType> & B,
    const TOpenCLMatrix<AFloat, AType> & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kIdentityDerivative,
                                             B, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Relu(TOpenCLMatrix<AFloat, AType> &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kRelu, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::ReluDerivative(      TOpenCLMatrix<AFloat, AType> & B,
                                            const TOpenCLMatrix<AFloat, AType> & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kReluDerivative, B, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Sigmoid(TOpenCLMatrix<AFloat, AType> &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kSigmoid, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::SigmoidDerivative(      TOpenCLMatrix<AFloat, AType> & B,
                                               const TOpenCLMatrix<AFloat, AType> & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kSigmoidDerivative,
                                             B, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Tanh(TOpenCLMatrix<AFloat, AType> &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kTanh, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::TanhDerivative(      TOpenCLMatrix<AFloat, AType> & B,
                                            const TOpenCLMatrix<AFloat, AType> & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kTanhDerivative,
                                             B, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::SymmetricRelu(TOpenCLMatrix<AFloat, AType> &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kSymmetricRelu, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::SymmetricReluDerivative(
          TOpenCLMatrix<AFloat, AType> & B,
    const TOpenCLMatrix<AFloat, AType> & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kSymmetricReluDerivative,
                                             B, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::SoftSign(TOpenCLMatrix<AFloat, AType> &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kSoftSign, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::SoftSignDerivative(
          TOpenCLMatrix<AFloat, AType> & B,
    const TOpenCLMatrix<AFloat, AType> & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kSoftSignDerivative,
                                             B, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Gauss(TOpenCLMatrix<AFloat, AType> &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kGauss, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::GaussDerivative(      TOpenCLMatrix<AFloat, AType> & B,
                                             const TOpenCLMatrix<AFloat, AType> & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kGaussDerivative, B, A);
}

} // namespace DNN
} // namespace TMVA
