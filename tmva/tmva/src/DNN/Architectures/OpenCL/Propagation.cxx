// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 02/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the functions required for the forward and //
 // backward propagation of activations through a neural network //
 // for OpenCL architectures.                                    //
 //////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"
#include "TMVA/DNN/Architectures/OpenCL/clBlas.h"

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::MultiplyTranspose(
          TOpenCLMatrix<AFloat, AType> & C,
    const TOpenCLMatrix<AFloat, AType> & A,
    const TOpenCLMatrix<AFloat, AType> & B)
{
   int m = A.GetNrows();
   int k = A.GetNcols();
   int n = B.GetNrows();

   cl_int error;
   error =  clblasSetup();
   cl_command_queue queue = A.GetComputeQueue()();
   cl_event event = NULL;

   error = gemm(clblasColumnMajor, clblasNoTrans, clblasTrans,
                m, n, k, static_cast<AFloat>(1.0),
                A.GetElementBuffer()(), 0, m,
                B.GetElementBuffer()(), 0, n, static_cast<AFloat>(0.0),
                C.GetElementBuffer()(), 0, m,
                1, &queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
   B.SetComputeQueue(A.GetComputeQueue());
   C.SetComputeQueue(A.GetComputeQueue());
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::AddRowWise(
          TOpenCLMatrix<AFloat, AType> &B,
    const TOpenCLMatrix<AFloat, AType> &A)
{
   const TOpenCLDevice<AFloat, AType> &device = B.GetDevice();

   int m     = (int) B.GetNrows();
   int n     = (int) B.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);

   cl::CommandQueue queue = B.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kAddRowWise, queue,
                        global, local, B.GetElementBuffer(),
                        A.GetElementBuffer(), m);
   A.SetComputeQueue(B.GetComputeQueue());
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Backward(
          TOpenCLMatrix<AFloat, AType> & activation_gradients_backward,
          TOpenCLMatrix<AFloat, AType> & weight_gradients,
          TOpenCLMatrix<AFloat, AType> & bias_gradients,
          TOpenCLMatrix<AFloat, AType> & df,
    const TOpenCLMatrix<AFloat, AType> & activation_gradients,
    const TOpenCLMatrix<AFloat, AType> & weights,
    const TOpenCLMatrix<AFloat, AType> & activation_backward)
{
   // Compute element-wise product.
   TOpenCL<AFloat, AType>::Hadamard(df, activation_gradients);

   // Activation gradients.
   if (activation_gradients_backward.GetNoElements() > 0) {
      TOpenCL<AFloat, AType>::Multiply(activation_gradients_backward, df, weights);
   }

   // Weight gradients.
   if (weight_gradients.GetNoElements() > 0) {
      TOpenCL<AFloat, AType>::TransposeMultiply(weight_gradients,
                                                df, activation_backward);
   }

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0) {
      TOpenCL<AFloat, AType>::SumColumns(bias_gradients, df);
   }
}

} // namespace DNN
} // namespace TMVA
