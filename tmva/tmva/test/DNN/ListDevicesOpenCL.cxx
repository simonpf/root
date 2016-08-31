// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 27/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


////////////////////////////////////////////////////////////////
// Platform independent device query app to check accessible  //
// devices.                                                   //
////////////////////////////////////////////////////////////////

#include "CL/cl.hpp"
#include <iostream>

int main()
{
   std::vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);

   std::cout << "Found " << platforms.size() << " platforms:" << std::endl;
   size_t index = 1;
   for (auto s : platforms) {
      std::cout << index << ":\t" << s.getInfo<CL_PLATFORM_NAME>() << " "
                << "Version: " << s.getInfo<CL_PLATFORM_VERSION>() << std::endl;
      std::cout << "  \tDevices: ";

      cl_context_properties contextProperties[3] = {
         CL_CONTEXT_PLATFORM, (cl_context_properties) s(), 0
      };
      cl::Context context(CL_DEVICE_TYPE_ALL, contextProperties);
      std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
      size_t deviceIndex = 1;
      for (auto d : devices) {
         if (deviceIndex > 1) {
            std::cout << "  \t         ";
         }
         std::cout << d.getInfo<CL_DEVICE_NAME>()
                   << ", OpenCL Version: " << d.getInfo<CL_DEVICE_VERSION>()
                   << std::endl << std::endl;
         deviceIndex++;
      }
      index++;
   }
}


