// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 28/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Overloaded wrappers for gemm and axpy routines from the clBLAS  //
// library that resolve the precision type and forward the call to //
// the corresponding function in the library.                      //
/////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_OPENCL_CLBLAS
#define TMVA_DNN_ARCHITECTURES_OPENCL_CLBLAS

#include "clBLAS.h"

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
/** Single precision matrix multiplication. Fowarded to clblasSgemm. */
clblasStatus gemm(clblasOrder order,
                  clblasTranspose transA,
                  clblasTranspose transB,
                  size_t M, size_t N, size_t K,
                  Real_t alpha, const cl_mem A, size_t offA, size_t lda,
                  const cl_mem B, size_t offB, size_t ldb, Real_t beta,
                  cl_mem C, size_t offC, size_t ldc,
                  cl_uint numCommandQueues, cl_command_queue * queues,
                  cl_uint numEventsInWaitList, const cl_event * list,
                  cl_event * event)
{
    clblasSgemm(order, transA, transB, M, N, K,
                alpha, A, offA, lda, B, offB, ldb, beta,
                C, offC, ldc,
                numCommandQueues, queues,
                numEventsInWaitList, list,
                event);
}

//____________________________________________________________________________
/** Double precision matrix multiplication. Fowarded to clblasDgemm. */
clblasStatus gemm(clblasOrder order,
                  clblasTranspose transA,
                  clblasTranspose transB,
                  size_t M, size_t N, size_t K,
                  Double_t alpha, const cl_mem A, size_t offA, size_t lda,
                  const cl_mem B, size_t offB, size_t ldb, Double_t beta,
                  cl_mem C, size_t offC, size_t ldc,
                  cl_uint numCommandQueues, cl_command_queue * queues,
                  cl_uint numEventsInWaitList, const cl_event * list,
                  cl_event * event)
{
    clblasDgemm(order, transA, transB, M, N, K,
                alpha, A, offA, lda, B, offB, ldb, beta,
                C, offC, ldc,
                numCommandQueues, queues,
                numEventsInWaitList, list,
                event);
}

//____________________________________________________________________________
/** Single precision vector addition. Fowarded to clblasSaxpy. */
clblasStatus axpy(size_t N, Real_t alpha,
                  const cl_mem X, size_t offx, size_t incx,
                  cl_mem Y, size_t offy, int incy,
                  cl_uint numCommandQueues, cl_command_queue * commandQueues,
                  cl_uint numEventsInWaitList, const cl_event * events,
                  cl_event * event)
{
    clblasSaxpy(N, alpha, X, offx, incx, Y, offy, incy,
                numCommandQueues, commandQueues,
                numEventsInWaitList, events, event);
}

//____________________________________________________________________________
/** Double precision vector addition. Fowarded to clblasDaxpy. */
clblasStatus axpy(size_t N, Double_t alpha,
                  const cl_mem X, size_t offx, size_t incx,
                  cl_mem Y, size_t offy, int incy,
                  cl_uint numCommandQueues, cl_command_queue * commandQueues,
                  cl_uint numEventsInWaitList, const cl_event * events,
                  cl_event * event)
{
    clblasDaxpy(N, alpha, X, offx, incx, Y, offy, incy,
                numCommandQueues, commandQueues,
                numEventsInWaitList, events, event);
}

} // namespace DNN
} // namespace TMVA

#endif
