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

inline OpenCLDouble_t ExecuteRegularizationKernel(EOpenCLKernel kernel,
                                                  const TOpenCLMatrix & A)
{

   const TOpenCLDevice &device = A.GetDevice();
   cl::CommandQueue     queue  = A.GetComputeQueue();

   try {

      int m     = (int) A.GetNrows();
      int n     = (int) A.GetNcols();

      cl::Buffer result(device.GetContext(), CL_MEM_WRITE_ONLY, sizeof(OpenCLDouble_t));
      cl::Buffer temp(device.GetContext(), CL_MEM_READ_WRITE, n * sizeof(OpenCLDouble_t));
      cl::LocalSpaceArg shared = cl::Local(device.localSize * sizeof(OpenCLDouble_t));

      cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
      cl::NDRange local(1, TOpenCLDevice::localSize);

      device.EnqueueKernel(kernel, queue, global, local,
                           A.GetElementBuffer(), m, shared, temp);

      global = cl::NDRange(1, TOpenCLDevice::localSize);
      device.EnqueueKernel(EOpenCLKernel::kSumVector, queue, global,
                           local, result, n, temp, shared);
      OpenCLDouble_t * hostResult =
         (OpenCLDouble_t *) queue.enqueueMapBuffer(result, CL_TRUE, CL_MAP_READ,
                                                   0, sizeof(OpenCLDouble_t));
      return hostResult[0];
   } catch (cl::Error error) {
      std::cout << "Error executing regularization kernel: "
                << device.GetErrorString(error.err()) << std::endl;
   }
}

OpenCLDouble_t TOpenCL::L1Regularization(const TOpenCLMatrix &A)
{
   return ExecuteRegularizationKernel(EOpenCLKernel::kL1Regularization, A);
}

void TOpenCL::AddL1RegularizationGradients(TOpenCLMatrix & B,
                                           const TOpenCLMatrix & A,
                                           OpenCLDouble_t weightDecay)
{
   const TOpenCLDevice &device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);

   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kAddL1RegularizationGradients,
                        queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(),
                        m, weightDecay);
   B.SetComputeQueue(queue);
}

OpenCLDouble_t TOpenCL::L2Regularization(const TOpenCLMatrix &A)
{
   return ExecuteRegularizationKernel(EOpenCLKernel::kL2Regularization, A);
}

void TOpenCL::AddL2RegularizationGradients(TOpenCLMatrix & B,
                                           const TOpenCLMatrix & A,
                                           OpenCLDouble_t weightDecay)
{
   const TOpenCLDevice &device = A.GetDevice();

   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);

   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kAddL2RegularizationGradients,
                        queue, global, local,
                        B.GetElementBuffer(), A.GetElementBuffer(),
                        m, weightDecay);
   B.SetComputeQueue(queue);
}

} // namespace DNN
} // namespace TMVA

