// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 03/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/DNN/Architectures/OpenCL.h"

//////////////////////////////////////////////////////////////////////
// Implementation of the Dropout function for OpenCL architectures. //
//////////////////////////////////////////////////////////////////////

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
void TOpenCL<AFloat, AType>::Dropout(TOpenCLMatrix<AFloat, AType> &A,
                                     AFloat dropoutProbability)
{
   const TOpenCLDevice<AFloat, AType> &device = A.GetDevice();
   int m     = (int) A.GetNrows();
   int n     = (int) A.GetNcols();

   cl::NDRange global(static_cast<size_t>(n),
                      TOpenCLDevice<AFloat, AType>::localSize);
   cl::NDRange local(1, TOpenCLDevice<AFloat, AType>::localSize);

   cl::CommandQueue queue = A.GetComputeQueue();
   device.EnqueueKernel(EOpenCLKernel::kDropout, queue,
                        global, local, A.GetElementBuffer(),
                        A.GetRandomStreamBuffer(), m, dropoutProbability);
}

} // namespace DNN
} // namespace TMVA
