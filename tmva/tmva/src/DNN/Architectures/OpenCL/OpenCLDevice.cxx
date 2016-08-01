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

#include "fstream"
#include "TMVA/DNN/Architectures/OpenCL/OpenCLMatrix.h"

#define STRINGIFY(s) STRING(s)
#define STRING(s) # s

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
   CompileKernels();
}

void TOpenCLDevice::CompileKernels()
{
   cl_int error;

   const char * filename = STRINGIFY(__KERNEL_FILE__);
   std::fstream file(filename);
   std::string source(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>{});

   const char * sourceString = source.c_str();
   size_t       sourceSize   = source.size();

   cl_program prog = clCreateProgramWithSource(fContext, 1,
                                               &sourceString, nullptr,
                                               &error);
   error = clBuildProgram(prog, 1, &fDeviceId, nullptr, nullptr, nullptr);
   HandleError(error);
   if (error != CL_SUCCESS) {
      PrintBuildError(prog);
   }

   fKernels[0] = clCreateKernel(prog, "Hadamard",                  &error);
   HandleError(error);
   fKernels[1] = clCreateKernel(prog, "SumColumns",                &error);
   HandleError(error);
   fKernels[2] = clCreateKernel(prog, "SumVector",                 &error);
   HandleError(error);
   fKernels[3] = clCreateKernel(prog, "SquaredErrorColumns",       &error);
   HandleError(error);
   fKernels[4] = clCreateKernel(prog, "MeanSquaredErrorGradients", &error);
   HandleError(error);
   fKernels[5] = clCreateKernel(prog, "CrossEntropyColumns",       &error);
   HandleError(error);
   fKernels[6] = clCreateKernel(prog, "CrossEntropyGradients",     &error);
   HandleError(error);

   // Activation Functions.
   fKernels[7] = clCreateKernel(prog, "IdentityDerivative",        &error);
   HandleError(error);
   fKernels[8] = clCreateKernel(prog, "Relu",                      &error);
   HandleError(error);
   fKernels[9] = clCreateKernel(prog, "ReluDerivative",            &error);
   HandleError(error);
   fKernels[10] = clCreateKernel(prog, "Sigmoid",                  &error);
   HandleError(error);
   fKernels[11] = clCreateKernel(prog, "SigmoidDerivative",        &error);
   HandleError(error);
   fKernels[12] = clCreateKernel(prog, "Tanh",                     &error);
   HandleError(error);
   fKernels[13] = clCreateKernel(prog, "TanhDerivative",           &error);
   HandleError(error);
   fKernels[14] = clCreateKernel(prog, "SymmetricRelu",            &error);
   HandleError(error);
   fKernels[15] = clCreateKernel(prog, "SymmetricReluDerivative",  &error);
   HandleError(error);
   fKernels[16] = clCreateKernel(prog, "SoftSign",                 &error);
   HandleError(error);
   fKernels[17] = clCreateKernel(prog, "SoftSignDerivative",       &error);
   HandleError(error);
   fKernels[18] = clCreateKernel(prog, "Gauss",                    &error);
   HandleError(error);
   fKernels[19] = clCreateKernel(prog, "GaussDerivative",          &error);
   HandleError(error);

   // Regularization Functions.
   fKernels[20] = clCreateKernel(prog, "L1RegularizationColumns",      &error);
   HandleError(error);
   fKernels[21] = clCreateKernel(prog, "AddL1RegularizationGradients", &error);
   HandleError(error);
   fKernels[22] = clCreateKernel(prog, "L2RegularizationColumns",      &error);
   HandleError(error);
   fKernels[23] = clCreateKernel(prog, "AddL2RegularizationGradients", &error);
   HandleError(error);
}

} // namespace DNN
} // namespace TMVA
