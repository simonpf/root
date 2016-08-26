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
#include "clBLAS.h"

namespace TMVA {
namespace DNN  {

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
   cl_command_queue queue = A.GetComputeQueue()();
   cl_event event = NULL;

   error = clblasDgemm(clblasColumnMajor, clblasNoTrans, clblasTrans,
                       m, n, k, 1.0,
                       A.GetElementBuffer()(), 0, m,
                       B.GetElementBuffer()(), 0, n, 0.0,
                       C.GetElementBuffer()(), 0, m,
                       1, &queue, 0, NULL, &event);
   A.GetDevice().HandleError(error);
   C.SetComputeQueue(A.GetComputeQueue());
}

//____________________________________________________________________________
void TOpenCL::AddRowWise(TOpenCLMatrix &B,
                         const TOpenCLMatrix &A)
{
   const TOpenCLDevice &device = B.GetDevice();

   int m     = (int) B.GetNrows();
   int n     = (int) B.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);

   cl::CommandQueue queue = B.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kAddRowWise, queue,
                        global, local, B.GetElementBuffer(),
                        A.GetElementBuffer(), m);
   B.SetComputeQueue(queue);
}

//____________________________________________________________________________
void TOpenCL::Backward(TOpenCLMatrix & activation_gradients_backward,
                       TOpenCLMatrix & weight_gradients,
                       TOpenCLMatrix & bias_gradients,
                       TOpenCLMatrix & df,
                       const TOpenCLMatrix & activation_gradients,
                       const TOpenCLMatrix & weights,
                       const TOpenCLMatrix & activation_backward)
{
   // Compute element-wise product.
   TOpenCL::Hadamard(df, activation_gradients);

   // Activation gradients.
   if (activation_gradients_backward.GetNoElements() > 0) {
      TOpenCL::Multiply(activation_gradients_backward, df, weights);
   }

   // Weight gradients.
   if (weight_gradients.GetNoElements() > 0) {
      TOpenCL::TransposeMultiply(weight_gradients, df, activation_backward);
   }

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0) {
      TOpenCL::SumColumns(bias_gradients, df);
   }
}

} // namespace DNN
} // namespace TMVA
