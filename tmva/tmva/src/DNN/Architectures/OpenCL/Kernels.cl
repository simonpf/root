// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 29/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Contains the OpenCL kernels for the implementation of the DNN //
// backend.                                                      //
///////////////////////////////////////////////////////////////////

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#include "clRNG/lfsr113.clh"

//
// Reduction Functions
//____________________________________________________________________________
AFloat SumWorkGroup(__local AFloat *sdata)
{
   int localSize  = get_local_size(0) * get_local_size(1);
   int localIndex = get_local_id(0) * get_local_size(1) + get_local_id(1);

   for (int offset = localSize / 2; offset > 0; offset /= 2) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (localIndex < offset) {
         sdata[localIndex] += sdata[localIndex + offset];
      }
   }
   return sdata[localIndex];
}

//____________________________________________________________________________
__kernel void Hadamard(__global AFloat *B,
                       __global const AFloat *A,
                       int m, int n)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      B[i] *= A[i];
   }
}

//____________________________________________________________________________
__kernel void SumColumns(__global AFloat *B,
                         __global const AFloat  *A,
                         int m, int n,
                         __local AFloat *sdata)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   AFloat sum = 0.0;
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      sum += A[i];
   }
   sdata[localIndexY] = sum;

   AFloat totalSum = SumWorkGroup(sdata);
   if (localIndexY == 0) {
      B[globalIndexX] = totalSum;
   }
}

//____________________________________________________________________________
__kernel void SumVector(__global AFloat * result,
                        int n,
                        __global AFloat * v,
                        __local  AFloat * sdata)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   if (globalIndexX == 0) {
      AFloat sum = 0.0;
      for (int i = localIndexY; i < n; i += localSizeY) {
         sum += v[i];
      }
      sdata[localIndexY] = sum;
      AFloat totalSum = SumWorkGroup(sdata);
      if (localIndexY == 0) {
         result[0] = totalSum;
      }
   }
}

//
// Propagation
//____________________________________________________________________________
__kernel void AddRowWise(__global AFloat * B,
                         __global const AFloat * A,
                         int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   AFloat sum = 0.0;
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
       B[i] += A[globalIndexX];
   }
}

//
// Loss Functions
//____________________________________________________________________________
__kernel void SquaredErrorColumns(__global const AFloat * A,
                                  __global const AFloat * B,
                                  int m,
                                  __local  AFloat *sdata,
                                  __global AFloat *gsdata)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   AFloat sum = 0.0;
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      AFloat err = B[i] - A[i];
      sum += err * err;
   }
   sdata[localIndexY] = sum;

   AFloat totalSum = SumWorkGroup(sdata);
   if (localIndexY == 0) {
      gsdata[globalIndexX] = totalSum;
   }
}

//____________________________________________________________________________
__kernel void CrossEntropyColumns(__global const AFloat * A,
                                  __global const AFloat * B,
                                  int m,
                                  __local  AFloat *sdata,
                                  __global AFloat *gsdata)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   AFloat sum = 0.0;
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      AFloat sig = 1.0 / (1.0 + exp(-B[i]));
      AFloat y   = A[i];
      sum -= y * log(sig) + (1.0 - y) * log(1.0 - sig);
   }
   sdata[localIndexY] = sum;

   AFloat totalSum = SumWorkGroup(sdata);
   if (localIndexY == 0) {
      gsdata[globalIndexX] = totalSum;
   }
}
//____________________________________________________________________________
__kernel void MeanSquaredErrorGradients(__global AFloat * C,
                                        __global const AFloat * A,
                                        __global const AFloat * B,
                                        int m, int n)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   AFloat norm = 1.0 / ((AFloat) (m * n));
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      C[i] = 2.0 * norm * (B[i] - A[i]);
   }
}

//____________________________________________________________________________
__kernel void CrossEntropyGradients(__global AFloat * C,
                                    __global const AFloat * A,
                                    __global const AFloat * B,
                                    int m, int n)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   AFloat norm = 1.0 / ((AFloat) (m * n));
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      AFloat sig = 1.0 / (1.0 + exp(-B[i]));
      AFloat y   = A[i];
      C[i] = norm * (sig - y);
   }
}

//
// Copy
//
//____________________________________________________________________________
__kernel void Copy(__global AFloat * B,
                   __global const AFloat * A,
                   int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
     B[i] = A[i];
   }
}

//
// Activation Functions
//
//____________________________________________________________________________
__kernel void IdentityDerivative(__global AFloat * B,
                                 __global const AFloat * A,
                                  int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
     B[i] = 1.0;
   }
}

//____________________________________________________________________________
__kernel void Relu(__global AFloat * A, int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      A[i] = (A[i] < 0.0) ? 0.0 : A[i];
   }
}

//____________________________________________________________________________
__kernel void ReluDerivative(__global AFloat * B,
                             __global const AFloat * A,
                             int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      B[i] = (A[i] < 0.0) ? 0.0 : 1.0;
   }
}


//____________________________________________________________________________
__kernel void Sigmoid(__global AFloat * A, int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      A[i] = 1.0 / (1.0 + exp(-A[i]));
   }
}

//____________________________________________________________________________
__kernel void SigmoidDerivative(__global AFloat * B,
                                __global const AFloat * A,
                                int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      AFloat sig = 1.0 / (1.0 + exp(-A[i]));
      B[i] = sig * (1.0 - sig);
   }
}

//____________________________________________________________________________
__kernel void Tanh(__global AFloat * A, int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      A[i] = tanh(A[i]);
   }
}

//____________________________________________________________________________
__kernel void TanhDerivative(__global AFloat * B,
                             __global const AFloat * A,
                             int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      AFloat t = tanh(A[i]);
      B[i] = 1.0 - t*t;
   }
}

//____________________________________________________________________________
__kernel void SymmetricRelu(__global AFloat * A, int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      A[i] = fabs(A[i]);
   }
}

//____________________________________________________________________________
__kernel void SymmetricReluDerivative(__global AFloat * B,
                                      __global const AFloat * A,
                                      int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      B[i] = (A[i] < 0.0) ? -1.0 : 1.0;
   }
}

//____________________________________________________________________________
__kernel void SoftSign(__global AFloat * A, int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      A[i] = A[i] / (1.0 + fabs(A[i]));
   }
}

//____________________________________________________________________________
__kernel void SoftSignDerivative(__global AFloat * B,
                                 __global const AFloat * A,
                                 int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      AFloat t = 1.0 + fabs(A[i]);
      B[i] = 1 / (t * t);
   }
}

//____________________________________________________________________________
__kernel void Gauss(__global AFloat * A, int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      A[i] = exp(- A[i] * A[i]);
   }
}

//____________________________________________________________________________
__kernel void GaussDerivative(__global AFloat * B,
                              __global const AFloat * A,
                              int m)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      AFloat t = exp(-A[i] * A[i]);
      B[i] = -2.0 * A[i] * t;
   }
}

//
// Regularization Functions
//
//____________________________________________________________________________
__kernel void L1RegularizationColumns(__global const AFloat * A,
                                      int m,
                                      __local  AFloat *sdata,
                                      __global AFloat *gsdata)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   AFloat sum = 0.0;
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      sum += fabs(A[i]);
   }
   sdata[localIndexY] = sum;

   AFloat totalSum = SumWorkGroup(sdata);
   if (localIndexY == 0) {
      gsdata[globalIndexX] = totalSum;
   }
}

__kernel void AddL1RegularizationGradients(__global AFloat * B,
                                           __global const AFloat * A,
                                           int m, AFloat weightDecay)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      B[i] += (A[i] < 0.0) ? -weightDecay : weightDecay;
   }
}

//____________________________________________________________________________
__kernel void L2RegularizationColumns(__global const AFloat * A,
                                      int m,
                                      __local  AFloat *sdata,
                                      __global AFloat *gsdata)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   AFloat sum = 0.0;
   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      sum += A[i] * A[i];
   }
   sdata[localIndexY] = sum;

   AFloat totalSum = SumWorkGroup(sdata);
   if (localIndexY == 0) {
      gsdata[globalIndexX] = totalSum;
   }
}

//____________________________________________________________________________
__kernel void AddL2RegularizationGradients(__global AFloat * B,
                                           __global const AFloat * A,
                                           int m, AFloat weightDecay)
{
   int globalIndexX = get_global_id(0);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      B[i] += 2.0 * weightDecay * A[i];
   }
}

//____________________________________________________________________________
__kernel void Dropout(__global AFloat * B,
                      __global clrngLfsr113HostStream * streams,
                      int m, AFloat dropoutProbability)
{
   int globalIndexX = get_global_id(0);
   int globalIndex  = globalIndexX * get_global_size(1) + get_global_id(1);
   int localIndexY  = get_local_id(1);
   int localSizeY   = get_local_size(1);

   int offset     = globalIndexX * m;

   clrngLfsr113Stream privateStream;
   clrngLfsr113CopyOverStreamsFromGlobal(1, &privateStream, streams + globalIndex);

   for (int i = offset + localIndexY; i < offset + m; i += localSizeY) {
      AFloat random = clrngLfsr113RandomU01(&privateStream);
      if (random < dropoutProbability) {
         B[i] /= dropoutProbability;
      } else {
         B[i] = 0.0;
      }
   }
}

