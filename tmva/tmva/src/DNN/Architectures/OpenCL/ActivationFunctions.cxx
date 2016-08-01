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
inline void ExecuteActivationFunctionKernel(cl_kernel kernel,
                                            TOpenCLMatrix & A)
{
   cl_int error;
   TOpenCLDevice &device = A.GetDevice();

   cl_mem Ad = A.GetElementPointer();
   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   error = clSetKernelArg(kernel, 0, sizeof(OpenCLDouble_t *), &Ad);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 1, sizeof(int), &m);

   size_t globalWorkSize[2] = {static_cast<size_t>(n), TOpenCLDevice::localSize};
   size_t localWorkSize[2]  = {1, TOpenCLDevice::localSize};
   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);
}

/** Launch an OpenCL kernel that applies an activation function derivative kernel
 *  to the two matrices by launching one workgroup per column */
//____________________________________________________________________________
inline void ExecuteActivationFunctionDerivativeKernel(
    cl_kernel kernel,
    TOpenCLMatrix & B,
    const TOpenCLMatrix &A)
{
   cl_int error;
   TOpenCLDevice &device = A.GetDevice();

   cl_mem Ad = A.GetElementPointer();
   cl_mem Bd = B.GetElementPointer();
   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   error = clSetKernelArg(kernel, 0, sizeof(OpenCLDouble_t *), &Bd);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 1, sizeof(OpenCLDouble_t *), &Ad);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 2, sizeof(int), &m);

   size_t globalWorkSize[2] = {static_cast<size_t>(n), TOpenCLDevice::localSize};
   size_t localWorkSize[2]  = {1, TOpenCLDevice::localSize};
   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);
}

//____________________________________________________________________________
void TOpenCL::Identity(TOpenCLMatrix & /*A*/) {}

//____________________________________________________________________________
void TOpenCL::IdentityDerivative(TOpenCLMatrix &B)
{
   cl_kernel kernel = B.GetDevice().GetKernel(EOpenCLKernel::kIdentityDerivative);
   ExecuteActivationFunctionKernel(kernel, B);
}

//____________________________________________________________________________
void TOpenCL::Relu(TOpenCLMatrix &A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kRelu);
   ExecuteActivationFunctionKernel(kernel, A);
}

//____________________________________________________________________________
void TOpenCL::ReluDerivative(      TOpenCLMatrix & B,
                             const TOpenCLMatrix & A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kReluDerivative);
   ExecuteActivationFunctionDerivativeKernel(kernel, B, A);
}

//____________________________________________________________________________
void TOpenCL::Sigmoid(TOpenCLMatrix &A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kSigmoid);
   ExecuteActivationFunctionKernel(kernel, A);
}

//____________________________________________________________________________
void TOpenCL::SigmoidDerivative(      TOpenCLMatrix & B,
                                const TOpenCLMatrix & A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kSigmoidDerivative);
   ExecuteActivationFunctionDerivativeKernel(kernel, B, A);
}

//____________________________________________________________________________
void TOpenCL::Tanh(TOpenCLMatrix &A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kTanh);
   ExecuteActivationFunctionKernel(kernel, A);
}

//____________________________________________________________________________
void TOpenCL::TanhDerivative(      TOpenCLMatrix & B,
                             const TOpenCLMatrix & A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kTanhDerivative);
   ExecuteActivationFunctionDerivativeKernel(kernel, B, A);
}

//____________________________________________________________________________
void TOpenCL::SymmetricRelu(TOpenCLMatrix &A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kSymmetricRelu);
   ExecuteActivationFunctionKernel(kernel, A);
}

//____________________________________________________________________________
void TOpenCL::SymmetricReluDerivative(      TOpenCLMatrix & B,
                                      const TOpenCLMatrix & A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kSymmetricReluDerivative);
   ExecuteActivationFunctionDerivativeKernel(kernel, B, A);
}

//____________________________________________________________________________
void TOpenCL::SoftSign(TOpenCLMatrix &A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kSoftSign);
   ExecuteActivationFunctionKernel(kernel, A);
}

//____________________________________________________________________________
void TOpenCL::SoftSignDerivative(      TOpenCLMatrix & B,
                                 const TOpenCLMatrix & A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kSoftSignDerivative);
   ExecuteActivationFunctionDerivativeKernel(kernel, B, A);
}

//____________________________________________________________________________
void TOpenCL::Gauss(TOpenCLMatrix &A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kGauss);
   ExecuteActivationFunctionKernel(kernel, A);
}

//____________________________________________________________________________
void TOpenCL::GaussDerivative(      TOpenCLMatrix & B,
                                 const TOpenCLMatrix & A)
{
   cl_kernel kernel = A.GetDevice().GetKernel(EOpenCLKernel::kGaussDerivative);
   ExecuteActivationFunctionDerivativeKernel(kernel, B, A);
}

} // namespace DNN
} // namespace TMVA
