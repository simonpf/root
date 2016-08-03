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
inline void ExecuteActivationFunctionKernel(EOpenCLKernel kernel,
                                            TOpenCLMatrix & A)
{
   TOpenCLDevice &device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);

   device.EnqueueKernel(kernel, global, local, A.GetElementBuffer(), m);
}

/** Launch an OpenCL kernel that applies an activation function derivative kernel
 *  to the two matrices by launching one workgroup per column */
//____________________________________________________________________________
inline void ExecuteActivationFunctionDerivativeKernel(
    EOpenCLKernel kernel,
    TOpenCLMatrix & B,
    const TOpenCLMatrix &A)
{
   TOpenCLDevice &device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);
   device.EnqueueKernel(kernel, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(), m);
}

//____________________________________________________________________________
void TOpenCL::Identity(TOpenCLMatrix & /*A*/) {}

//____________________________________________________________________________
void TOpenCL::IdentityDerivative(TOpenCLMatrix &B)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kIdentityDerivative, B);
}

//____________________________________________________________________________
void TOpenCL::Relu(TOpenCLMatrix &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kRelu, A);
}

//____________________________________________________________________________
void TOpenCL::ReluDerivative(      TOpenCLMatrix & B,
                             const TOpenCLMatrix & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kReluDerivative, B, A);
}

//____________________________________________________________________________
void TOpenCL::Sigmoid(TOpenCLMatrix &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kSigmoid, A);
}

//____________________________________________________________________________
void TOpenCL::SigmoidDerivative(      TOpenCLMatrix & B,
                                const TOpenCLMatrix & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kSigmoidDerivative,
                                             B, A);
}

//____________________________________________________________________________
void TOpenCL::Tanh(TOpenCLMatrix &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kTanh, A);
}

//____________________________________________________________________________
void TOpenCL::TanhDerivative(      TOpenCLMatrix & B,
                             const TOpenCLMatrix & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kTanhDerivative,
                                             B, A);
}

//____________________________________________________________________________
void TOpenCL::SymmetricRelu(TOpenCLMatrix &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kSymmetricRelu, A);
}

//____________________________________________________________________________
void TOpenCL::SymmetricReluDerivative(      TOpenCLMatrix & B,
                                      const TOpenCLMatrix & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kSymmetricReluDerivative,
                                             B, A);
}

//____________________________________________________________________________
void TOpenCL::SoftSign(TOpenCLMatrix &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kSoftSign, A);
}

//____________________________________________________________________________
void TOpenCL::SoftSignDerivative(      TOpenCLMatrix & B,
                                 const TOpenCLMatrix & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kSoftSignDerivative,
                                             B, A);
}

//____________________________________________________________________________
void TOpenCL::Gauss(TOpenCLMatrix &A)
{
   ExecuteActivationFunctionKernel(EOpenCLKernel::kGauss, A);
}

//____________________________________________________________________________
void TOpenCL::GaussDerivative(      TOpenCLMatrix & B,
                                 const TOpenCLMatrix & A)
{
   ExecuteActivationFunctionDerivativeKernel(EOpenCLKernel::kGaussDerivative, B, A);
}

} // namespace DNN
} // namespace TMVA
