// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 05/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Implementation of the specialization of the CopyBatch member //
// functions of the TDeviceDataLoader class for OpenCL          //
// architectures.                                               //
//////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"

namespace TMVA {
namespace DNN  {

// Static Members
std::shared_ptr<TOpenCLDevice> TOpenCL::fDefaultDevice{};

std::shared_ptr<TOpenCLDevice> TOpenCL::GetDefaultDevice()
{
   if (fDefaultDevice) {
      return fDefaultDevice;
   } else {
      fDefaultDevice = std::shared_ptr<TOpenCLDevice>(new TOpenCLDevice);
      return fDefaultDevice;
   }
}

} // namespace TMVA
} // namespace DNN
