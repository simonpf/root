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

inline OpenCLDouble_t ExecuteLossFunctionsKernel(EOpenCLKernel kernel,
                                                 const TOpenCLMatrix & Y,
                                                 const TOpenCLMatrix & output)
{
   const TOpenCLDevice &device = Y.GetDevice();
   cl::CommandQueue     queue  = Y.GetComputeQueue();

   try {
      int m     = (int) Y.GetNrows();
      int n     = (int) Y.GetNcols();

      cl::Buffer result(device.GetContext(), CL_MEM_WRITE_ONLY, sizeof(OpenCLDouble_t));
      cl::Buffer temp(device.GetContext(), CL_MEM_READ_WRITE, n * sizeof(OpenCLDouble_t));
      cl::LocalSpaceArg shared = cl::Local(device.localSize * sizeof(OpenCLDouble_t));

      cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
      cl::NDRange local(1, TOpenCLDevice::localSize);

      device.EnqueueKernel(kernel, queue, global, local,
                           Y.GetElementBuffer(), output.GetElementBuffer(),
                           m, shared, temp);

      global = cl::NDRange(1, TOpenCLDevice::localSize);
      device.EnqueueKernel(EOpenCLKernel::kSumVector, queue,
                           global, local, result, n, temp, shared);
      OpenCLDouble_t * hostResult
          = (OpenCLDouble_t *) queue.enqueueMapBuffer(result, CL_TRUE, CL_MAP_READ,
                                                      0, sizeof(OpenCLDouble_t));
      OpenCLDouble_t norm = 1.0 / static_cast<OpenCLDouble_t>(m * n);
      return norm * hostResult[0];

   } catch (cl::Error error) {
      std::cout << "Error executing loss function kernel: "
                << device.GetErrorString(error.err()) << std::endl;
   }
}

OpenCLDouble_t TOpenCL::MeanSquaredError(const TOpenCLMatrix & Y,
                                         const TOpenCLMatrix & output)
{
   return ExecuteLossFunctionsKernel(EOpenCLKernel::kSquaredErrorColumns, Y, output);
}

void TOpenCL::MeanSquaredErrorGradients(TOpenCLMatrix &dY,
                                        const TOpenCLMatrix & Y,
                                        const TOpenCLMatrix & output)
{
   const TOpenCLDevice &device = Y.GetDevice();

   int m     = (int) Y.GetNrows();
   int n     = (int) Y.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);
   cl::CommandQueue queue = Y.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kMeanSquaredErrorGradients,
                        queue, global, local,
                        dY.GetElementBuffer(), Y.GetElementBuffer(),
                        output.GetElementBuffer(), m, n);
   dY.SetComputeQueue(queue);
}

OpenCLDouble_t TOpenCL::CrossEntropy(const TOpenCLMatrix & Y,
                                     const TOpenCLMatrix & output)
{
   return ExecuteLossFunctionsKernel(EOpenCLKernel::kCrossEntropyColumns, Y, output);
}

void TOpenCL::CrossEntropyGradients(TOpenCLMatrix &dY,
                                    const TOpenCLMatrix & Y,
                                    const TOpenCLMatrix & output)
{
   const TOpenCLDevice &device = Y.GetDevice();

   int m     = (int) Y.GetNrows();
   int n     = (int) Y.GetNcols();

   cl::NDRange global(static_cast<size_t>(n), TOpenCLDevice::localSize);
   cl::NDRange local(1, TOpenCLDevice::localSize);
   cl::CommandQueue queue = Y.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kCrossEntropyGradients,
                        queue, global, local,
                        dY.GetElementBuffer(), Y.GetElementBuffer(),
                        output.GetElementBuffer(), m, n);
   dY.SetComputeQueue(queue);
}

} // namespace DNN
} // namespace TMVA
