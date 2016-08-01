// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 01/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////
// Regularization functions for the OpenCL implementation of         //
// deep neural networks.                                             //
///////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"

namespace TMVA {
namespace DNN  {

inline OpenCLDouble_t ExecuteRegularizationKernel(cl_kernel kernel,
                                                  const TOpenCLMatrix & A)
{
   cl_int error;
   TOpenCLDevice &device = A.GetDevice();
   cl_kernel kernelSumVector  = device.GetKernel(EOpenCLKernel::kSumVector);

   cl_mem Ad = A.GetElementPointer();
   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl_mem result, temp;
   result = clCreateBuffer(device.GetContext(), CL_MEM_WRITE_ONLY,
                           sizeof(OpenCLDouble_t), nullptr, & error);
   temp   = clCreateBuffer(device.GetContext(), CL_MEM_READ_WRITE,
                           n * sizeof(OpenCLDouble_t), nullptr, & error);

   error = clSetKernelArg(kernel, 0, sizeof(OpenCLDouble_t *), &Ad);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 1, sizeof(int), &m);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 2,
                          TOpenCLDevice::localSize * sizeof(OpenCLDouble_t),
                          nullptr);
   error = clSetKernelArg(kernel, 3, sizeof(OpenCLDouble_t *), &temp);
   device.HandleError(error);

   size_t globalWorkSize[2] = {static_cast<size_t>(n), TOpenCLDevice::localSize};
   size_t localWorkSize[2]  = {1, TOpenCLDevice::localSize};
   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);

   error = clSetKernelArg(kernelSumVector, 0, sizeof(OpenCLDouble_t *), &result);
   device.HandleError(error);
   error = clSetKernelArg(kernelSumVector, 1, sizeof(int), &n);
   device.HandleError(error);
   error = clSetKernelArg(kernelSumVector, 2, sizeof(OpenCLDouble_t *), &temp);
   device.HandleError(error);
   error = clSetKernelArg(kernelSumVector, 3,
                          TOpenCLDevice::localSize * sizeof(OpenCLDouble_t),
                          nullptr);
   device.HandleError(error);
   globalWorkSize[0] = 1;
   clEnqueueNDRangeKernel(device.GetQueue(), kernelSumVector,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);

   OpenCLDouble_t * hostResult =
       (OpenCLDouble_t*) clEnqueueMapBuffer(device.GetQueue(), result,
                                            CL_TRUE, CL_MAP_READ, 0,
                                            sizeof(OpenCLDouble_t),
                                            0, nullptr, nullptr, &error);
   device.HandleError(error);
   return hostResult[0];
}

OpenCLDouble_t TOpenCL::L1Regularization(const TOpenCLMatrix &A)
{
   cl_kernel kernel =
       A.GetDevice().GetKernel(EOpenCLKernel::kL1Regularization);
   return ExecuteRegularizationKernel(kernel, A);

}

void TOpenCL::AddL1RegularizationGradients(TOpenCLMatrix & B,
                                           const TOpenCLMatrix & A,
                                           OpenCLDouble_t weightDecay)
{
   cl_int error;
   TOpenCLDevice &device = A.GetDevice();
   cl_kernel kernel = device.GetKernel(EOpenCLKernel::kAddL1RegularizationGradients);

   cl_mem Ad = A.GetElementPointer();
   cl_mem Bd = B.GetElementPointer();
   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   error = clSetKernelArg(kernel, 0, sizeof(OpenCLDouble_t *), &Bd);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 1, sizeof(OpenCLDouble_t *), &Ad);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 2, sizeof(int), &m);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 3, sizeof(double), &weightDecay);
   device.HandleError(error);

   size_t globalWorkSize[2] = {static_cast<size_t>(n), TOpenCLDevice::localSize};
   size_t localWorkSize[2]  = {1, TOpenCLDevice::localSize};
   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);

}

OpenCLDouble_t TOpenCL::L2Regularization(const TOpenCLMatrix &A)
{
   cl_kernel kernel =
       A.GetDevice().GetKernel(EOpenCLKernel::kL2Regularization);
   return ExecuteRegularizationKernel(kernel, A);
}

void TOpenCL::AddL2RegularizationGradients(TOpenCLMatrix & B,
                                           const TOpenCLMatrix & A,
                                           OpenCLDouble_t weightDecay)
{
   cl_int error;
   TOpenCLDevice &device = A.GetDevice();
   cl_kernel kernel = device.GetKernel(EOpenCLKernel::kAddL2RegularizationGradients);

   cl_mem Ad = A.GetElementPointer();
   cl_mem Bd = B.GetElementPointer();
   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   error = clSetKernelArg(kernel, 0, sizeof(OpenCLDouble_t *), &Bd);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 1, sizeof(OpenCLDouble_t *), &Ad);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 2, sizeof(int), &m);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 3, sizeof(double), &weightDecay);
   device.HandleError(error);

   size_t globalWorkSize[2] = {static_cast<size_t>(n), TOpenCLDevice::localSize};
   size_t localWorkSize[2]  = {1, TOpenCLDevice::localSize};
   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);

}

} // namespace DNN
} // namespace TMVA

