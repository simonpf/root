// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 27/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////
// Implementation of OpenCLMatrix class member functions. //
////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL/OpenCLMatrix.h"

namespace TMVA {
namespace DNN  {

TOpenCLDevice::TOpenCLDevice()
{
   cl_int err;

   clGetPlatformIDs(1, &fPlatform, nullptr);
   clGetDeviceIDs(fPlatform, CL_DEVICE_TYPE_DEFAULT, 1, &fDeviceId, nullptr);

   fProperties[1] = (cl_context_properties) fPlatform;
   fContext       = clCreateContext(fProperties, 1, &fDeviceId,
                                    nullptr, nullptr, &err);
   HandleError(err);
   fQueue         = clCreateCommandQueue(fContext, fDeviceId, 0, &err);
   HandleError(err);
}

} // namespace DNN
} // namespace TMVA
