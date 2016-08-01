// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 31/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////
// Loss functions for OpenCL implementation of deep neural //
// networks.                                               //
/////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"

namespace TMVA {
namespace DNN  {

inline OpenCLDouble_t ExecuteLossFunctionsKernel(cl_kernel kernel,
                                                 const TOpenCLMatrix & Y,
                                                 const TOpenCLMatrix & output)
{
   cl_int error;
   TOpenCLDevice &device = Y.GetDevice();
   cl_kernel kernelSumVector  = device.GetKernel(EOpenCLKernel::kSumVector);

   cl_mem Ad = Y.GetElementPointer();
   cl_mem Bd = output.GetElementPointer();
   int m     = (int) Y.GetNrows();
   int n     = (int) Y.GetNcols();

   cl_mem result, temp;
   result = clCreateBuffer(device.GetContext(), CL_MEM_WRITE_ONLY,
                           sizeof(OpenCLDouble_t), nullptr, & error);
   temp   = clCreateBuffer(device.GetContext(), CL_MEM_READ_WRITE,
                           n * sizeof(OpenCLDouble_t), nullptr, & error);

   error = clSetKernelArg(kernel, 0, sizeof(OpenCLDouble_t *), &Ad);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 1, sizeof(OpenCLDouble_t *), &Bd);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 2, sizeof(int), &m);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 3,
                          TOpenCLDevice::localSize * sizeof(OpenCLDouble_t),
                          nullptr);
   error = clSetKernelArg(kernel, 4, sizeof(OpenCLDouble_t *), &temp);
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
   OpenCLDouble_t norm = 1.0 / static_cast<OpenCLDouble_t>(m * n);
   return norm * hostResult[0];
}

OpenCLDouble_t TOpenCL::MeanSquaredError(const TOpenCLMatrix & Y,
                                         const TOpenCLMatrix & output)
{
   cl_kernel kernel =
       Y.GetDevice().GetKernel(EOpenCLKernel::kSquaredErrorColumns);
   return ExecuteLossFunctionsKernel(kernel, Y, output);
}

void TOpenCL::MeanSquaredErrorGradients(TOpenCLMatrix &dY,
                                        const TOpenCLMatrix & Y,
                                        const TOpenCLMatrix & output)
{
   cl_int error;
   TOpenCLDevice &device = Y.GetDevice();
   cl_kernel kernel = device.GetKernel(EOpenCLKernel::kMeanSquaredErrorGradients);

   cl_mem Ad = Y.GetElementPointer();
   cl_mem Bd = output.GetElementPointer();
   cl_mem Cd = dY.GetElementPointer();
   int m     = (int) Y.GetNrows();
   int n     = (int) Y.GetNcols();

   error = clSetKernelArg(kernel, 0, sizeof(OpenCLDouble_t *), &Cd);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 1, sizeof(OpenCLDouble_t *), &Ad);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 2, sizeof(OpenCLDouble_t *), &Bd);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 3, sizeof(int), &m);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 4, sizeof(int), &n);
   device.HandleError(error);

   size_t globalWorkSize[2] = {static_cast<size_t>(n), TOpenCLDevice::localSize};
   size_t localWorkSize[2]  = {1, TOpenCLDevice::localSize};
   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);
}

OpenCLDouble_t TOpenCL::CrossEntropy(const TOpenCLMatrix & Y,
                                     const TOpenCLMatrix & output)
{
   cl_kernel kernel =
       Y.GetDevice().GetKernel(EOpenCLKernel::kCrossEntropyColumns);
   return ExecuteLossFunctionsKernel(kernel, Y, output);
}

void TOpenCL::CrossEntropyGradients(TOpenCLMatrix &dY,
                                    const TOpenCLMatrix & Y,
                                    const TOpenCLMatrix & output)
{
   cl_int error;
   TOpenCLDevice &device = Y.GetDevice();
   cl_kernel kernel = device.GetKernel(EOpenCLKernel::kCrossEntropyGradients);

   cl_mem Ad = Y.GetElementPointer();
   cl_mem Bd = output.GetElementPointer();
   cl_mem Cd = dY.GetElementPointer();
   int m     = (int) Y.GetNrows();
   int n     = (int) Y.GetNcols();

   error = clSetKernelArg(kernel, 0, sizeof(OpenCLDouble_t *), &Cd);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 1, sizeof(OpenCLDouble_t *), &Ad);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 2, sizeof(OpenCLDouble_t *), &Bd);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 3, sizeof(int), &m);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 4, sizeof(int), &n);
   device.HandleError(error);

   size_t globalWorkSize[2] = {static_cast<size_t>(n), TOpenCLDevice::localSize};
   size_t localWorkSize[2]  = {1, TOpenCLDevice::localSize};
   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);
}

} // namespace DNN
} // namespace TMVA
