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
#include "OpenCL/ActivationFunctions.cxx"
#include "OpenCL/Arithmetic.cxx"
#include "OpenCL/Dropout.cxx"
#include "OpenCL/Initialization.cxx"
#include "OpenCL/LossFunctions.cxx"
#include "OpenCL/Propagation.cxx"
#include "OpenCL/Regularization.cxx"

namespace TMVA {
namespace DNN  {

// Static Members
template<typename AFloat, EOpenCLDeviceType AType>
std::shared_ptr<TOpenCLDevice<AFloat, AType>> TOpenCL<AFloat, AType>::fDefaultDevice{};

template<typename AFloat, EOpenCLDeviceType AType>
auto TOpenCL<AFloat, AType>::GetDefaultDevice()
    -> std::shared_ptr<TOpenCLDevice<AFloat, AType>>
{
   if (fDefaultDevice) {
      return fDefaultDevice;
   } else {
      fDefaultDevice = std::shared_ptr<TOpenCLDevice<AFloat, AType>>(
          new TOpenCLDevice<AFloat, AType>);
      return fDefaultDevice;
   }
}

template class TOpenCL<Real_t,   EOpenCLDeviceType::kGpu>;
template class TOpenCL<Double_t, EOpenCLDeviceType::kGpu>;

} // namespace TMVA
} // namespace DNN
