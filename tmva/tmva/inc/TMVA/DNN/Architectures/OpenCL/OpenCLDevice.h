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

#define __CL_ENABLE_EXCEPTIONS
#define __CL_VERSION_1_2

#include "CL/cl.hpp"
#include <vector>
#include <tuple>

namespace TMVA {
namespace DNN  {

class TOpenCLMatrix;

enum class EOpenCLKernel : int {
   kHadamard                  = 0,
   kSumColumns                = 1,
   kSumVector                 = 2,

   // Propagation.
   kAddRowWise                = 3,

   // Loss Functions.
   kSquaredErrorColumns       = 4,
   kMeanSquaredErrorGradients = 5,
   kCrossEntropyColumns       = 6,
   kCrossEntropyGradients     = 7,

   // Activation Functions.
   kIdentityDerivative        = 8,
   kRelu                      = 9,
   kReluDerivative            = 10,
   kSigmoid                   = 11,
   kSigmoidDerivative         = 12,
   kTanh                      = 13,
   kTanhDerivative            = 14,
   kSymmetricRelu             = 15,
   kSymmetricReluDerivative   = 16,
   kSoftSign                  = 17,
   kSoftSignDerivative        = 18,
   kGauss                     = 19,
   kGaussDerivative           = 20,

   // Regularization
   kL1Regularization             = 21,
   kAddL1RegularizationGradients = 22,
   kL2Regularization             = 23,
   kAddL2RegularizationGradients = 24,

   // Dropout
   kDropout = 25
};

class TOpenCLDevice
{
private:

   cl::Device       fDevice;
   cl::Context      fContext       = 0;
   cl::CommandQueue fQueue         = 0;
   cl::Program      fProgram;
   cl::Kernel       fKernels[26];

public:

   static constexpr size_t localSizeX = 16;
   static constexpr size_t localSizeY = 16;
   static constexpr size_t localSize  = localSizeX * localSizeY;

   TOpenCLDevice();
   inline void HandleError(cl_int error) const;
   inline const char * GetErrorString(cl_int error) const;
   inline void PrintBuildError(cl::Program program) const;

   cl::Context      GetContext() const {return fContext;}
   cl::CommandQueue GetQueue()   const {return fQueue;}
   cl::Device       GetDevice()  const {return fDevice;}

   template<typename ...Args>
   inline void EnqueueKernel(EOpenCLKernel kernelEnum,
                             cl::NDRange globalSize,
                             cl::NDRange localSize,
                             Args ...args);

private:

   void CompileKernels();


   template<
      typename TupleType,
      int size = std::tuple_size<TupleType>::value,
      int index = 0
   >
   struct SetArguments {
      static inline void execute(cl::Kernel kernel, const TupleType &arguments) {
          kernel.setArg(index, std::get<index>(arguments));
          SetArguments<TupleType, size, index+1>::execute(kernel, arguments);
       }
   };

   template<typename TupleType, int size>
   struct SetArguments<TupleType, size, size> {
      static inline void execute(cl::Kernel, const TupleType &) {}
   };

};

inline void TOpenCLDevice::HandleError(cl_int error) const
{
   if (error != CL_SUCCESS) {
      std::cout << "OpenCL Device Error:"
                << GetErrorString(error) << std::endl;
   }
}

inline void TOpenCLDevice::PrintBuildError(cl::Program program) const
{
   std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(fDevice);
   std::cout << log << std::endl;
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

template <class ...Args>
inline void TOpenCLDevice::EnqueueKernel(EOpenCLKernel kernelEnum,
                                         cl::NDRange global,
                                         cl::NDRange local,
                                         Args ...args)
{
   cl::Kernel kernel = fKernels[static_cast<int>(kernelEnum)];
   auto arguments = std::make_tuple(args...);

   SetArguments<decltype(arguments)>::execute(kernel, arguments);
   fQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
}

} // namespace DNN
} // namespace TMVA

#endif
