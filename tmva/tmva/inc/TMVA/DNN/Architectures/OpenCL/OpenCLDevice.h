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

#include "Rtypes.h"
#include "CL/cl.hpp"
#include <vector>
#include <tuple>
#include <iostream>

namespace TMVA {
namespace DNN  {

/** Enum class represting OpenCL device types. */
enum class EOpenCLDeviceType : int {
    kAccelerator = CL_DEVICE_TYPE_ACCELERATOR,
    kCpu         = CL_DEVICE_TYPE_CPU,
    kDefault     = CL_DEVICE_TYPE_DEFAULT,
    kGpu         = CL_DEVICE_TYPE_GPU
};

/** Enum class representing the OpenCL kernels requires
 *  to implement the low-level interface. */
enum class EOpenCLKernel : int {

   // Arithmetic.
   kHadamard                  = 0,
   kSumColumns                = 1,
   kSumVector                 = 2,

   // Propagation.
   kAddRowWise                = 3,

   // Copy.
   kCopy                      = 4,

   // Loss Functions.
   kSquaredErrorColumns       = 5,
   kMeanSquaredErrorGradients = 6,
   kCrossEntropyColumns       = 7,
   kCrossEntropyGradients     = 8,

   // Activation Functions.
   kIdentityDerivative        = 9,
   kRelu                      = 10,
   kReluDerivative            = 11,
   kSigmoid                   = 12,
   kSigmoidDerivative         = 13,
   kTanh                      = 14,
   kTanhDerivative            = 15,
   kSymmetricRelu             = 16,
   kSymmetricReluDerivative   = 17,
   kSoftSign                  = 18,
   kSoftSignDerivative        = 19,
   kGauss                     = 20,
   kGaussDerivative           = 21,

   // Regularization
   kL1Regularization             = 22,
   kAddL1RegularizationGradients = 23,
   kL2Regularization             = 24,
   kAddL2RegularizationGradients = 25,

   // Dropout
   kDropout = 26
};

/** Helper struct converting a floating pointer number type to the corresponding
*  OpenCL C data type as a pointer to const char. */
template<typename T>
struct TypeName;

template<>
struct TypeName<Real_t>
{
    static constexpr const char * value = "float";
};

template<>
struct TypeName<Double_t>
{
    static constexpr const char * value = "double";
};

template <typename AFloat, EOpenCLDeviceType AType>
class TOpenCLMatrix;

//____________________________________________________________________________
//
// OpenCL Device Class
//____________________________________________________________________________
/** Helper class representing OpenCL devices. This class takes care of
 *  initializing the OpenCL device as well as compiling the kernels. Also
 *  provides helper functions for error handling and enqueueing of kernels.
 *
 * \tparam AFloat Floating point type to represent matrix elements on the device.
 * Supported types are Double_t and Real_t, which are mapped to the double precision
 * and single precision floating point types on the device.
 * \tparam ADeviceType The type of the OpenCL device represented by an
 * EOpenCLDevice enum.
 */
template<typename AFloat, EOpenCLDeviceType AType = EOpenCLDeviceType::kGpu>
class TOpenCLDevice
{
private:

   cl::Device       fDevice;
   cl::Context      fContext;
   cl::Program      fProgram;
   cl::Kernel       fKernels[27];
   cl::CommandQueue fDefaultQueue;

   size_t fNComputeQueues;

public:

   static constexpr size_t localSizeX = 16;
   static constexpr size_t localSizeY = 16;
   static constexpr size_t localSize  = localSizeX * localSizeY;

   TOpenCLDevice();
   TOpenCLDevice(const TOpenCLDevice  &)             = default;
   TOpenCLDevice(      TOpenCLDevice &&)             = default;
   TOpenCLDevice & operator=(const TOpenCLDevice  &) = default;
   TOpenCLDevice & operator=(      TOpenCLDevice &&) = default;

   inline void HandleError(cl_int error) const;
   inline const char * GetErrorString(cl_int error) const;
   inline void PrintBuildError(cl::Program program) const;

   cl::Context      GetContext() const {return fContext;}
   cl::CommandQueue GetQueue()   const {return fDefaultQueue;}
   cl::Device       GetDevice()  const {return fDevice;}

   /** Variadic function that sets OpenCL kernel arguments. Executes a
    *  compile-time loop that calls setArg for each of the arguments with
    *  the corresponding argument index. The executes the kernel on the
    *  provided command queue and the provided global and local grids,
    *  \p globalSize and \p localSize. */
   template<typename ...Args>
   inline void EnqueueKernel(EOpenCLKernel kernelEnum,
                             cl::CommandQueue,
                             cl::NDRange globalSize,
                             cl::NDRange localSize,
                             Args ...args) const;

private:

   void CompileKernels();

   /** Helper struct for that implements a compile-time loop over a tuple
    *  of kernel arguments.*/
   template
   <
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

//
// Inline Functions.
//

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
inline void TOpenCLDevice<AFloat, AType>::HandleError(cl_int error) const
{
   if (error != CL_SUCCESS) {
      std::cout << "OpenCL Device Error:"
                << GetErrorString(error) << std::endl;
   }
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
inline void TOpenCLDevice<AFloat, AType>::PrintBuildError(cl::Program program) const
{
   std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(fDevice);
   std::cout << log << std::endl;
}

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
inline auto TOpenCLDevice<AFloat, AType>::GetErrorString(cl_int error) const
    -> const char *
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

//____________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
template <class ...Args>
inline void TOpenCLDevice<AFloat, AType>::EnqueueKernel(
    EOpenCLKernel kernelEnum,
    cl::CommandQueue queue,
    cl::NDRange global,
    cl::NDRange local,
    Args ...args) const
{
   cl::Kernel kernel = fKernels[static_cast<int>(kernelEnum)];
   auto arguments = std::make_tuple(args...);

   try {
      SetArguments<decltype(arguments)>::execute(kernel, arguments);
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
   } catch (cl::Error error) {
      HandleError(error.err());
   }

}

} // namespace DNN
} // namespace TMVA

#endif
