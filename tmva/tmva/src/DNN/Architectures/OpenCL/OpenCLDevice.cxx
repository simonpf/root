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

// TOpenCLDeviceBuffer
//____________________________________________________________________________
TOpenCLDevice::TOpenCLDeviceBuffer::TOpenCLDeviceBuffer(size_t size,
                                                        const TOpenCLDevice & device)
    : fDevice(device)
{
   fBuffer = cl::Buffer(fDevice.GetContext(), CL_MEM_READ_WRITE,
                        size * sizeof(OpenCLDouble_t));
}

// TOpenCLDeviceBuffer
//____________________________________________________________________________
TOpenCLDevice::TOpenCLDeviceBuffer::TOpenCLDeviceBuffer(cl::Buffer buffer,
                                                        const TOpenCLDevice & device)
    : fBuffer(buffer), fDevice(device)
{
   // Nothing to do here.
}

//____________________________________________________________________________
auto TOpenCLDevice::TOpenCLDeviceBuffer::GetSubBuffer(size_t start,
                                                      size_t size)
    -> TOpenCLDeviceBuffer
{
   cl_int error;
   _cl_buffer_region region;
   region.origin = start * sizeof(OpenCLDouble_t);
   region.size   = size  * sizeof(OpenCLDouble_t);

   cl::Buffer subBuffer = fBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                  CL_BUFFER_CREATE_TYPE_REGION,
                                                  (void *) &region,
                                                  &error);
   return TOpenCLDeviceBuffer(subBuffer, fDevice);
}

//____________________________________________________________________________
void TOpenCLDevice::TOpenCLDeviceBuffer::SetUnconsumed()
{
   fConsumptionEvent = clCreateUserEvent(fDevice.GetContext()(), NULL);
}

//____________________________________________________________________________
void TOpenCLDevice::TOpenCLDeviceBuffer::SetConsumed()
{
   clSetUserEventStatus(fConsumptionEvent(), CL_COMPLETE);
}

//____________________________________________________________________________
void TOpenCLDevice::TOpenCLDeviceBuffer::SynchronizeComputation() const
{
   if (fConsumptionEvent()) {
      fConsumptionEvent.wait();
   }
}

//____________________________________________________________________________
void TOpenCLDevice::TOpenCLDeviceBuffer::SynchronizeTransfer() const
{
   fTransferEvent.wait();
}

// TOpenCLHostBuffer
//____________________________________________________________________________
TOpenCLDevice::TOpenCLHostBuffer::TOpenCLHostBuffer(size_t size,
                                                    cl::Context context,
                                                    cl::CommandQueue dataQueue)
    : fContext(context), fDataQueue(dataQueue), fSize(size)
{
    fBuffer = cl::Buffer(fContext, CL_MEM_ALLOC_HOST_PTR,
                         size * sizeof(OpenCLDouble_t));
    fDataPointer = (OpenCLDouble_t *) fDataQueue.enqueueMapBuffer(
        fBuffer,
        CL_TRUE,
        CL_MEM_READ_WRITE,
        0, size * sizeof(OpenCLDouble_t));
}

//____________________________________________________________________________
TOpenCLDevice::TOpenCLHostBuffer::TOpenCLHostBuffer(TOpenCLHostBuffer &&other)
    : fBuffer(std::move(other.fBuffer)), fContext(std::move(other.fContext)),
      fDataPointer(other.fDataPointer), fDataQueue(std::move(other.fDataQueue)),
      fSize(other.fSize)
{
   // Nothing to do here.
}

//____________________________________________________________________________
void TOpenCLDevice::TOpenCLHostBuffer::Lock()
{
   fLock.acquire(fMutex);
}

//____________________________________________________________________________
void TOpenCLDevice::TOpenCLHostBuffer::Release()
{
   fLock.release();
}

//____________________________________________________________________________
void TOpenCLDevice::TOpenCLHostBuffer::CopyTo(TOpenCLDeviceBuffer & deviceBuffer)
{
   cl::Event transferEvent;
   fDataQueue.enqueueWriteBuffer(deviceBuffer.GetBuffer(), CL_FALSE, 0,
                                 fSize * sizeof(OpenCLDouble_t), (void*) fDataPointer, nullptr, &transferEvent),
   deviceBuffer.SetTransferEvent(transferEvent);
}

//____________________________________________________________________________
OpenCLDouble_t & TOpenCLDevice::TOpenCLHostBuffer::operator[](size_t index)
{
   return fDataPointer[index];
}

//____________________________________________________________________________
OpenCLDouble_t TOpenCLDevice::TOpenCLHostBuffer::operator[](size_t index) const
{
   return fDataPointer[index];
}

// TOpenCLDevice
//____________________________________________________________________________
TOpenCLDevice::TOpenCLDevice(size_t nComputeQueues)
    : fDevice(), fContext(), fProgram(), fKernels(), fNComputeQueues(nComputeQueues)
{
   std::cout << "Constructing opencl device." << std::endl;
   try {
       std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      std::cout << "Found " << platforms.size() << " platforms: " << std::endl;
      for (size_t i = 0; i < platforms.size(); i++) {
         std::cout << "\tNAME: " << platforms[i].getInfo<CL_PLATFORM_NAME>()
                   << std::endl;
         std::cout << "\tVersion: " << platforms[i].getInfo<CL_PLATFORM_VERSION>()
                   << std::endl;
      }

      cl_context_properties contextProperties[3] = {
         CL_CONTEXT_PLATFORM,
         (cl_context_properties) (platforms[0])(),
         0
      };

      fContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, contextProperties);

      std::vector<cl::Device> devices = fContext.getInfo<CL_CONTEXT_DEVICES>();
      std::cout << "Found " << devices.size() << " devices: " << std::endl;
      for (size_t i = 0; i < devices.size(); i++) {
         std::cout << "\tNAME: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
         std::cout << "\tVersion: " << devices[i].getInfo<CL_DRIVER_VERSION>()
                   << std::endl << std::endl;
         std::cout << "\tVersion: " << devices[i].getInfo<CL_DEVICE_VERSION>()
                   << std::endl << std::endl;
      }
      fDevice = devices[0];

      for (size_t i = 0; i < fNComputeQueues; i++) {
         fQueues.push_back(cl::CommandQueue(fContext, fDevice));
      }

      CompileKernels();
   } catch(cl::Error error) {
      std::cout << "Error initializing OpenCL device: " << error.what() << std::endl;
   }
}

TOpenCLDevice::TOpenCLDeviceBuffer TOpenCLDevice::CreateDeviceBuffer(size_t size)
{
   return TOpenCLDeviceBuffer(size, *this);
}

TOpenCLDevice::TOpenCLHostBuffer TOpenCLDevice::CreateHostBuffer(size_t size)
{
   return TOpenCLHostBuffer(size, fContext, cl::CommandQueue(fContext, fDevice));
}

void TOpenCLDevice::CompileKernels()
{
   const char * filename = STRINGIFY(__KERNEL_FILE__);
   std::fstream file(filename);
   std::string source(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>{});

   std::vector<cl::Device> devices(1);
   devices[0] = fDevice;
   cl::Program program(fContext, source, false);

   try {

      std::cout << "Arguments: -I " STRINGIFY(__KERNEL_INCLUDE_DIRS__) << std::endl;
      program.build(devices, "-cl-nv-verbose -I " STRINGIFY(__KERNEL_INCLUDE_DIRS__));

      fKernels[0] = cl::Kernel(program, "Hadamard");
      fKernels[1] = cl::Kernel(program, "SumColumns");
      fKernels[2] = cl::Kernel(program, "SumVector");

      fKernels[3] = cl::Kernel(program, "AddRowWise");

      fKernels[4] = cl::Kernel(program, "SquaredErrorColumns");
      fKernels[5] = cl::Kernel(program, "MeanSquaredErrorGradients");
      fKernels[6] = cl::Kernel(program, "CrossEntropyColumns");
      fKernels[7] = cl::Kernel(program, "CrossEntropyGradients");

      // Activation Functions.
      fKernels[8]  = cl::Kernel(program, "IdentityDerivative");
      fKernels[9]  = cl::Kernel(program, "Relu");
      fKernels[10] = cl::Kernel(program, "ReluDerivative");
      fKernels[11] = cl::Kernel(program, "Sigmoid");
      fKernels[12] = cl::Kernel(program, "SigmoidDerivative");
      fKernels[13] = cl::Kernel(program, "Tanh");
      fKernels[14] = cl::Kernel(program, "TanhDerivative");
      fKernels[15] = cl::Kernel(program, "SymmetricRelu");
      fKernels[16] = cl::Kernel(program, "SymmetricReluDerivative");
      fKernels[17] = cl::Kernel(program, "SoftSign");
      fKernels[18] = cl::Kernel(program, "SoftSignDerivative");
      fKernels[19] = cl::Kernel(program, "Gauss");
      fKernels[20] = cl::Kernel(program, "GaussDerivative");

      // Regularization Functions.
      fKernels[21] = cl::Kernel(program, "L1RegularizationColumns");
      fKernels[22] = cl::Kernel(program, "AddL1RegularizationGradients");
      fKernels[23] = cl::Kernel(program, "L2RegularizationColumns");
      fKernels[24] = cl::Kernel(program, "AddL2RegularizationGradients");

      // Dropout.
      fKernels[25] = cl::Kernel(program, "Dropout");
   } catch (cl::Error error) {
      HandleError(error.err());
      std::string log;
      program.getBuildInfo(fDevice, CL_PROGRAM_BUILD_LOG, &log);
      std::cout << log << std::endl;
      std::vector<char *> binaries = program.getInfo<CL_PROGRAM_BINARIES>();
      std::vector<size_t> sizes    = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
      FILE *fp = std::fopen("binary.ptx", "w");
      std::fwrite(binaries[0], sizeof(char), sizes[0], fp);
      std::fclose(fp);
   }
}

} // namespace DNN
} // namespace TMVA
