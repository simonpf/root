// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 26/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Declaration of the TOpenCL Device class which represents OpenCL //
// devices.                                                        //
/////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_OPENCL_OPENCLDEVICE
#define TMVA_DNN_ARCHITECTURES_OPENCL_OPENCLDEVICE

#include "CL/cl.h"
#include <vector>

namespace TMVA {
namespace DNN  {

class TOpenCLMatrix;

enum class EOpenCLKernel : int {
   kHadamard                  = 0,
   kSumColumns                = 1,
   kSumVector                 = 2,
   kSquaredErrorColumns       = 3,
   kMeanSquaredErrorGradients = 4,
   kCrossEntropyColumns       = 5,
   kCrossEntropyGradients     = 6,

   // Activation Functions.
   kIdentityDerivative        = 7,
   kRelu                      = 8,
   kReluDerivative            = 9,
   kSigmoid                   = 10,
   kSigmoidDerivative         = 11,
   kTanh                      = 12,
   kTanhDerivative            = 13,
   kSymmetricRelu             = 14,
   kSymmetricReluDerivative   = 15,
   kSoftSign                  = 16,
   kSoftSignDerivative        = 17,
   kGauss                     = 18,
   kGaussDerivative           = 19,

   // Regularization
   kL1Regularization             = 20,
   kAddL1RegularizationGradients = 21,
   kL2Regularization             = 22,
   kAddL2RegularizationGradients = 23,
};

class TOpenCLDevice
{
private:

   cl_platform_id        fPlatform      = 0;
   cl_device_id          fDeviceId      = 0;
   cl_context_properties fProperties[3] = {CL_CONTEXT_PLATFORM, 0, 0};
   cl_context            fContext       = 0;
   cl_command_queue      fQueue         = 0;
   cl_program            fProgram;
   cl_kernel             fKernels[24];

public:

   static constexpr size_t localSizeX = 16;
   static constexpr size_t localSizeY = 16;
   static constexpr size_t localSize  = localSizeX * localSizeY;

   TOpenCLDevice();
   inline void HandleError(cl_int error) const;
   inline void PrintBuildError(cl_program program) const;

   cl_context         GetContext() const   {return   fContext;}
   cl_command_queue   GetQueue()           {return   fQueue;}
   cl_command_queue * GetQueuePointer()    {return & fQueue;}
   cl_device_id       GetDeviceId()        {return   fDeviceId;}
   cl_device_id *     GetDeviceIdPointer() {return & fDeviceId;}

   inline const cl_kernel & GetKernel(EOpenCLKernel kernel);
   inline void GetWorkSizes(size_t * globalWork, size_t * localWork,
                            size_t nCols, size_t nRows) const;

private:

   void CompileKernels();
   inline const char * GetErrorString(cl_int error) const;

};

inline void TOpenCLDevice::HandleError(cl_int error) const
{
   if (error != CL_SUCCESS) {
      std::cout << "OpenCL Device Error:"
                << GetErrorString(error) << std::endl;
   }
}

inline void TOpenCLDevice::PrintBuildError(cl_program program) const
{
   size_t len;
   clGetProgramBuildInfo(program, fDeviceId, CL_PROGRAM_BUILD_LOG,
                         0, nullptr, &len);
   std::vector<char> buffer(len);
   clGetProgramBuildInfo(program, fDeviceId, CL_PROGRAM_BUILD_LOG, len,
                         buffer.data(), nullptr);
   std::cout << buffer.data() << std::endl;
}

inline const char * TOpenCLDevice::GetErrorString(cl_int error) const
{
   switch(error){
   // run-time and JIT compiler errors
   case   0: return "CL_SUCCESS";
   case  -1: return "CL_DEVICE_NOT_FOUND";
   case  -2: return "CL_DEVICE_NOT_AVAILABLE";
   case  -3: return "CL_COMPILER_NOT_AVAILABLE";
   case  -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
   case  -5: return "CL_OUT_OF_RESOURCES";
   case  -6: return "CL_OUT_OF_HOST_MEMORY";
   case  -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
   case  -8: return "CL_MEM_COPY_OVERLAP";
   case  -9: return "CL_IMAGE_FORMAT_MISMATCH";
   case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
   case -11: return "CL_BUILD_PROGRAM_FAILURE";
   case -12: return "CL_MAP_FAILURE";
   case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
   case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
   case -15: return "CL_COMPILE_PROGRAM_FAILURE";
   case -16: return "CL_LINKER_NOT_AVAILABLE";
   case -17: return "CL_LINK_PROGRAM_FAILURE";
   case -18: return "CL_DEVICE_PARTITION_FAILED";
   case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

   // compile-time errors
   case -30: return "CL_INVALID_VALUE";
   case -31: return "CL_INVALID_DEVICE_TYPE";
   case -32: return "CL_INVALID_PLATFORM";
   case -33: return "CL_INVALID_DEVICE";
   case -34: return "CL_INVALID_CONTEXT";
   case -35: return "CL_INVALID_QUEUE_PROPERTIES";
   case -36: return "CL_INVALID_COMMAND_QUEUE";
   case -37: return "CL_INVALID_HOST_PTR";
   case -38: return "CL_INVALID_MEM_OBJECT";
   case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
   case -40: return "CL_INVALID_IMAGE_SIZE";
   case -41: return "CL_INVALID_SAMPLER";
   case -42: return "CL_INVALID_BINARY";
   case -43: return "CL_INVALID_BUILD_OPTIONS";
   case -44: return "CL_INVALID_PROGRAM";
   case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
   case -46: return "CL_INVALID_KERNEL_NAME";
   case -47: return "CL_INVALID_KERNEL_DEFINITION";
   case -48: return "CL_INVALID_KERNEL";
   case -49: return "CL_INVALID_ARG_INDEX";
   case -50: return "CL_INVALID_ARG_VALUE";
   case -51: return "CL_INVALID_ARG_SIZE";
   case -52: return "CL_INVALID_KERNEL_ARGS";
   case -53: return "CL_INVALID_WORK_DIMENSION";
   case -54: return "CL_INVALID_WORK_GROUP_SIZE";
   case -55: return "CL_INVALID_WORK_ITEM_SIZE";
   case -56: return "CL_INVALID_GLOBAL_OFFSET";
   case -57: return "CL_INVALID_EVENT_WAIT_LIST";
   case -58: return "CL_INVALID_EVENT";
   case -59: return "CL_INVALID_OPERATION";
   case -60: return "CL_INVALID_GL_OBJECT";
   case -61: return "CL_INVALID_BUFFER_SIZE";
   case -62: return "CL_INVALID_MIP_LEVEL";
   case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
   case -64: return "CL_INVALID_PROPERTY";
   case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
   case -66: return "CL_INVALID_COMPILER_OPTIONS";
   case -67: return "CL_INVALID_LINKER_OPTIONS";
   case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

   // extension errors
   case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
   case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
   case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
   case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
   case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
   case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
   default: return "Unknown OpenCL error";
   }
}

inline const cl_kernel & TOpenCLDevice::GetKernel(EOpenCLKernel kernel)
{
   return fKernels[static_cast<int>(kernel)];
}

inline void TOpenCLDevice::GetWorkSizes(size_t *globalWork,
                                        size_t *localWork,
                                        size_t nRows, size_t nCols) const
{
    localWork[0] = localSizeX;
    localWork[1] = localSizeY;

    size_t gridDimX = nCols / localSizeX;
    if (nCols % localSizeX != 0) {
        gridDimX += 1;
    }

    size_t gridDimY = nRows / localSizeY;
    if (nRows % localSizeY != 0) {
        gridDimY += 1;
    }

    globalWork[0] = localSizeX * gridDimX;
    globalWork[1] = localSizeY * gridDimY;
}

} // namespace DNN
} // namespace TMVA

#endif
