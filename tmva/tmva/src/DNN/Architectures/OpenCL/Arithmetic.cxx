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
   cl_command_queue queue = A.GetComputeQueue()();
   cl_event event = NULL;

   error = clblasDgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                     m, n, k, 1.0,
                     A.GetElementBuffer()(), 0, m,
                     B.GetElementBuffer()(), 0, k, 0.0,
                     C.GetElementBuffer()(), 0, m,
                     1, &queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
   C.SetComputeQueue(A.GetComputeQueue());
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
   cl_command_queue queue = A.GetComputeQueue()();
   cl_event event = NULL;

   error = clblasDgemm(clblasColumnMajor, clblasTrans, clblasNoTrans,
                     m, n, k, 1.0,
                     A.GetElementBuffer()(), 0, k,
                     B.GetElementBuffer()(), 0, k, 0.0,
                     C.GetElementBuffer()(), 0, m,
                     1, &queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
   C.SetComputeQueue(A.GetComputeQueue());
}

//____________________________________________________________________________
void TOpenCL::Hadamard(TOpenCLMatrix & B,
                       const TOpenCLMatrix & A)
{
   const TOpenCLDevice &device    = B.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange      global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange      local(1, TOpenCLDevice::localSize);
   cl::CommandQueue queue = A.GetComputeQueue();

   device.EnqueueKernel(EOpenCLKernel::kHadamard, queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(), m, n);
   B.SetComputeQueue(queue);
}

//____________________________________________________________________________
void TOpenCL::SumColumns(TOpenCLMatrix & B,
                         const TOpenCLMatrix & A)
{
   const TOpenCLDevice &device = B.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);
   cl::CommandQueue queue = A.GetComputeQueue();

   device.EnqueueKernel(EOpenCLKernel::kSumColumns, queue,
                        global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(), m, n,
                        cl::Local(TOpenCLDevice::localSize * sizeof(OpenCLDouble_t)));
   B.SetComputeQueue(queue);
}

//____________________________________________________________________________
void TOpenCL::ScaleAdd(TOpenCLMatrix & B,
                       const TOpenCLMatrix & A,
                       OpenCLDouble_t alpha)
{
   int m = A.GetNrows();
   int n = B.GetNcols();

   cl_int error;
   error =  clblasSetup();
   cl_command_queue queue = A.GetComputeQueue()();

   error = clblasDaxpy(n * m, alpha,
                       A.GetElementBuffer()(), 0, 1,
                       B.GetElementBuffer()(), 0, 1,
                       1, &queue, 0, NULL, NULL);
   B.SetComputeQueue(A.GetComputeQueue());
}

//____________________________________________________________________________
void TOpenCL::Copy(TOpenCLMatrix & B,
                   const TOpenCLMatrix & A)
{
   const TOpenCLDevice &device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);
   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kCopy, queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(), m);
}
} // namespace DNN
} // namespace TMVA
