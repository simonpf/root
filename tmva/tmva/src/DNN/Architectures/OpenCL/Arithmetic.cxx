// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 26/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////
// Arithmetic helper functions for the            //
// implementation of the OpenCL backend for DNNs. //
////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"
#include "clBLAS.h"
#include <fstream>
#include <string>

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
void TOpenCL::Multiply(TOpenCLMatrix & C,
                       const TOpenCLMatrix & A,
                       const TOpenCLMatrix & B)
{
   int m = A.GetNrows();
   int k = A.GetNcols();
   int n = B.GetNcols();

   cl_int error;
   error =  clblasSetup();
   cl_command_queue *queue = C.GetDevice().GetQueuePointer();
   cl_event event = NULL;

   error = clblasDgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                     m, n, k, 1.0,
                     A.GetElementPointer(), 0, m,
                     B.GetElementPointer(), 0, k, 0.0,
                     C.GetElementPointer(), 0, m,
                     1, queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
}

//____________________________________________________________________________
void TOpenCL::MultiplyTranspose(TOpenCLMatrix & C,
                                const TOpenCLMatrix & A,
                                const TOpenCLMatrix & B)
{
   int m = A.GetNrows();
   int k = A.GetNcols();
   int n = B.GetNrows();

   cl_int error;
   error =  clblasSetup();
   cl_command_queue *queue = C.GetDevice().GetQueuePointer();
   cl_event event = NULL;

   error = clblasDgemm(clblasColumnMajor, clblasNoTrans, clblasTrans,
                     m, n, k, 1.0,
                     A.GetElementPointer(), 0, m,
                     B.GetElementPointer(), 0, n, 0.0,
                     C.GetElementPointer(), 0, m,
                     1, queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
}

//____________________________________________________________________________
void TOpenCL::TransposeMultiply(TOpenCLMatrix & C,
                                const TOpenCLMatrix & A,
                                const TOpenCLMatrix & B)
{
   int m = A.GetNcols();
   int k = A.GetNrows();
   int n = B.GetNcols();

   cl_int error;
   error =  clblasSetup();
   cl_command_queue *queue = C.GetDevice().GetQueuePointer();
   cl_event event = NULL;

   error = clblasDgemm(clblasColumnMajor, clblasTrans, clblasNoTrans,
                     m, n, k, 1.0,
                     A.GetElementPointer(), 0, k,
                     B.GetElementPointer(), 0, k, 0.0,
                     C.GetElementPointer(), 0, m,
                     1, queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
}

//____________________________________________________________________________
void TOpenCL::Hadamard(TOpenCLMatrix & B,
                       const TOpenCLMatrix & A)
{
   cl_int error;
   TOpenCLDevice &device    = B.GetDevice();
   cl_kernel kernel = device.GetKernel(EOpenCLKernel::kHadamard);

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
   error = clSetKernelArg(kernel, 3, sizeof(int), &n);
   device.HandleError(error);

   size_t globalWorkSize[2];
   size_t localWorkSize[2];
   device.GetWorkSizes(globalWorkSize, localWorkSize, A.GetNrows(), A.GetNcols());

   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);
}

//____________________________________________________________________________
void TOpenCL::SumColumns(TOpenCLMatrix & B,
                         const TOpenCLMatrix & A)
{
   cl_int error;
   TOpenCLDevice &device = B.GetDevice();
   cl_kernel kernel = device.GetKernel(EOpenCLKernel::kSumColumns);

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
   error = clSetKernelArg(kernel, 3, sizeof(int), &n);
   device.HandleError(error);
   error = clSetKernelArg(kernel, 4,
                          TOpenCLDevice::localSize * sizeof(OpenCLDouble_t),
                          nullptr);
   device.HandleError(error);

   size_t globalWorkSize[2] = {A.GetNcols(), TOpenCLDevice::localSize};
   size_t localWorkSize[2]  = {1, TOpenCLDevice::localSize};

   clEnqueueNDRangeKernel(device.GetQueue(), kernel,
                          2, nullptr, globalWorkSize, localWorkSize,
                          0, nullptr, nullptr);
}

} // namespace DNN
} // namespace TMVA
