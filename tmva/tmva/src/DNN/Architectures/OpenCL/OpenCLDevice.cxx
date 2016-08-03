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
    : fDevice(), fContext(), fQueue(), fProgram(), fKernels()
{
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

      fQueue   = cl::CommandQueue(fContext, fDevice);
      CompileKernels();
   } catch(cl::Error error) {
      std::cout << "Error initializing OpenCL device: " << error.what() << std::endl;
   }
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
