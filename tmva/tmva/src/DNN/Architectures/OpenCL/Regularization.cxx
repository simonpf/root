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

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
inline AFloat ExecuteRegularizationKernel(      EOpenCLKernel kernel,
                                          const TOpenCLMatrix<AFloat, AType> & A)
{

   const TOpenCLDevice<AFloat, AType> & device = A.GetDevice();
   cl::CommandQueue                     queue  = A.GetComputeQueue();

   try {

      int m     = (int) A.GetNrows();
      int n     = (int) A.GetNcols();

      cl::Buffer result(device.GetContext(), CL_MEM_WRITE_ONLY, sizeof(AFloat));
      cl::Buffer temp(device.GetContext(), CL_MEM_READ_WRITE, n * sizeof(AFloat));
      cl::LocalSpaceArg shared = cl::Local(device.localSize * sizeof(AFloat));

      cl::NDRange global(static_cast<size_t>(n),
                         TOpenCLDevice<AFloat, AType>::localSize);
      cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);

      device.EnqueueKernel(kernel, queue, global, local,
                           A.GetElementBuffer(), m, shared, temp);

      global = cl::NDRange(1, TOpenCLDevice<AFloat, AType>::localSize);
      device.EnqueueKernel(EOpenCLKernel::kSumVector, queue, global,
                           local, result, n, temp, shared);
      AFloat * hostResult =
         (AFloat *) queue.enqueueMapBuffer(result, CL_TRUE, CL_MAP_READ,
                                                   0, sizeof(AFloat));
      return hostResult[0];
   } catch (cl::Error error) {
      std::cout << "Error executing regularization kernel: "
                << device.GetErrorString(error.err()) << std::endl;
   }
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
AFloat TOpenCL<AFloat, AType>::L1Regularization(
    const TOpenCLMatrix<AFloat, AType> & A)
{
   return ExecuteRegularizationKernel(EOpenCLKernel::kL1Regularization, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::AddL1RegularizationGradients(
          TOpenCLMatrix<AFloat, AType> & B,
    const TOpenCLMatrix<AFloat, AType> & A,
          AFloat weightDecay)
{
   const TOpenCLDevice<AFloat, AType> & device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);

   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kAddL1RegularizationGradients,
                        queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(),
                        m, weightDecay);
   B.SetComputeQueue(queue);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
AFloat TOpenCL<AFloat, AType>::L2Regularization(const TOpenCLMatrix<AFloat, AType> &A)
{
   return ExecuteRegularizationKernel(EOpenCLKernel::kL2Regularization, A);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::AddL2RegularizationGradients(
          TOpenCLMatrix<AFloat, AType> & B,
    const TOpenCLMatrix<AFloat, AType> & A,
          AFloat weightDecay)
{
   const TOpenCLDevice<AFloat, AType> & device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);

   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kAddL2RegularizationGradients,
                        queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(),
                        m, weightDecay);
   B.SetComputeQueue(queue);
}

} // namespace DNN
} // namespace TMVA
