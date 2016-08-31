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

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
inline AFloat ExecuteLossFunctionsKernel(
          EOpenCLKernel kernel,
    const TOpenCLMatrix<AFloat, AType> & Y,
    const TOpenCLMatrix<AFloat, AType> & output)
{
   const TOpenCLDevice<AFloat, AType> & device = Y.GetDevice();
   cl::CommandQueue                     queue  = output.GetComputeQueue();

   try {
      int m     = (int) Y.GetNrows();
      int n     = (int) Y.GetNcols();

      cl::Buffer result(device.GetContext(), CL_MEM_WRITE_ONLY, sizeof(AFloat));
      cl::Buffer temp(device.GetContext(), CL_MEM_READ_WRITE, n * sizeof(AFloat));
      cl::LocalSpaceArg shared = cl::Local(device.localSize * sizeof(AFloat));

      cl::NDRange global(static_cast<size_t>(n),
                         TOpenCLDevice<AFloat, AType>::localSize);
      cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);

      device.EnqueueKernel(kernel, queue, global, local,
                           Y.GetElementBuffer(), output.GetElementBuffer(),
                           m, shared, temp);

      global = cl::NDRange(1, TOpenCLDevice<AFloat, AType>::localSize);
      device.EnqueueKernel(EOpenCLKernel::kSumVector, queue,
                           global, local, result, n, temp, shared);
      AFloat * hostResult
          = (AFloat *) queue.enqueueMapBuffer(result, CL_TRUE, CL_MAP_READ,
                                                      0, sizeof(AFloat));
      AFloat norm = 1.0 / static_cast<AFloat>(m * n);
      return norm * hostResult[0];

   } catch (cl::Error error) {
      std::cout << "Error executing loss function kernel: "
                << device.GetErrorString(error.err()) << std::endl;
   }
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
AFloat TOpenCL<AFloat, AType>::MeanSquaredError(
    const TOpenCLMatrix<AFloat, AType> & Y,
    const TOpenCLMatrix<AFloat, AType> & output)
{
   return ExecuteLossFunctionsKernel(EOpenCLKernel::kSquaredErrorColumns, Y, output);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::MeanSquaredErrorGradients(
          TOpenCLMatrix<AFloat, AType> & dY,
    const TOpenCLMatrix<AFloat, AType> & Y,
    const TOpenCLMatrix<AFloat, AType> & output)
{
   const TOpenCLDevice<AFloat, AType> &device = Y.GetDevice();

   int m     = (int) Y.GetNrows();
   int n     = (int) Y.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);
   cl::CommandQueue queue = output.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kMeanSquaredErrorGradients,
                        queue, global, local,
                        dY.GetElementBuffer(), Y.GetElementBuffer(),
                        output.GetElementBuffer(), m, n);
   dY.SetComputeQueue(queue);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
AFloat TOpenCL<AFloat, AType>::CrossEntropy(
    const TOpenCLMatrix<AFloat, AType> & Y,
    const TOpenCLMatrix<AFloat, AType> & output)
{
   return ExecuteLossFunctionsKernel(EOpenCLKernel::kCrossEntropyColumns, Y, output);
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::CrossEntropyGradients(
          TOpenCLMatrix<AFloat, AType> & dY,
    const TOpenCLMatrix<AFloat, AType> & Y,
    const TOpenCLMatrix<AFloat, AType> & output)
{
   const TOpenCLDevice<AFloat, AType> &device = Y.GetDevice();

   int m     = (int) Y.GetNrows();
   int n     = (int) Y.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);
   cl::CommandQueue queue = output.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kCrossEntropyGradients,
                        queue, global, local,
                        dY.GetElementBuffer(), Y.GetElementBuffer(),
                        output.GetElementBuffer(), m, n);
   dY.SetComputeQueue(queue);
}

} // namespace DNN
} // namespace TMVA
