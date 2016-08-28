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
#include "TMVA/DNN/Architectures/OpenCL/clBlas.h"
#include <fstream>
#include <string>

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Multiply(      TOpenCLMatrix<AFloat, AType> & C,
                                      const TOpenCLMatrix<AFloat, AType> & A,
                                      const TOpenCLMatrix<AFloat, AType> & B)
{
   int m = A.GetNrows();
   int k = A.GetNcols();
   int n = B.GetNcols();

   cl_int error;
   error =  clblasSetup();
   cl_command_queue queue = A.GetComputeQueue()();
   cl_event event = NULL;

   error = gemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                m, n, k, static_cast<AFloat>(1.0),
                A.GetElementBuffer()(), 0, m,
                B.GetElementBuffer()(), 0, k, static_cast<AFloat>(0.0),
                C.GetElementBuffer()(), 0, m,
                1, &queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
   C.SetComputeQueue(A.GetComputeQueue());
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::TransposeMultiply(
          TOpenCLMatrix<AFloat, AType> & C,
    const TOpenCLMatrix<AFloat, AType> & A,
    const TOpenCLMatrix<AFloat, AType> & B)
{
   int m = A.GetNcols();
   int k = A.GetNrows();
   int n = B.GetNcols();

   cl_int error;
   error =  clblasSetup();
   cl_command_queue queue = B.GetComputeQueue()();
   cl_event event = NULL;

   error = gemm(clblasColumnMajor, clblasTrans, clblasNoTrans,
                m, n, k, static_cast<AFloat>(1.0),
                A.GetElementBuffer()(), 0, k,
                B.GetElementBuffer()(), 0, k, static_cast<AFloat>(0.0),
                C.GetElementBuffer()(), 0, m,
                1, &queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
   C.SetComputeQueue(B.GetComputeQueue());
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Hadamard(      TOpenCLMatrix<AFloat, AType> & B,
                                      const TOpenCLMatrix<AFloat, AType> & A)
{
   const TOpenCLDevice<AFloat, AType> &device = B.GetDevice();
   int m = (int) A.GetNrows();
   int n = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);
   cl::CommandQueue queue = A.GetComputeQueue();

   device.EnqueueKernel(EOpenCLKernel::kHadamard, queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(), m, n);
   B.SetComputeQueue(queue);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::SumColumns(      TOpenCLMatrix<AFloat, AType> & B,
                                        const TOpenCLMatrix<AFloat, AType> & A)
{
   const TOpenCLDevice<AFloat, AType> &device = B.GetDevice();
   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);
   cl::CommandQueue queue = A.GetComputeQueue();

   device.EnqueueKernel(EOpenCLKernel::kSumColumns, queue,
                        global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(), m, n,
                        cl::Local(TOpenCLDevice<AFloat, AType>::localSize * sizeof(AFloat)));
   B.SetComputeQueue(queue);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::ScaleAdd(      TOpenCLMatrix<AFloat, AType> & B,
                                      const TOpenCLMatrix<AFloat, AType> & A,
                                            AFloat alpha)
{
   int m = A.GetNrows();
   int n = B.GetNcols();

   cl_int error;
   error =  clblasSetup();
   cl_command_queue queue = A.GetComputeQueue()();

   error = axpy(n * m, alpha,
                A.GetElementBuffer()(), 0, 1,
                B.GetElementBuffer()(), 0, 1,
                1, &queue, 0, NULL, NULL);
   B.SetComputeQueue(A.GetComputeQueue());
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Copy(      TOpenCLMatrix<AFloat, AType> & B,
                                  const TOpenCLMatrix<AFloat, AType> & A)
{
   const TOpenCLDevice<AFloat, AType> &device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);
   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kCopy, queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(), m);
}
} // namespace DNN
} // namespace TMVA
