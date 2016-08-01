// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 26/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////
// Declaration of the TOpenCLMatrix class used for the //
// representation of matrices on OpenCL devices.       //
/////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_OPENCL_OPENCLMATRIX
#define TMVA_DNN_ARCHITECTURES_OPENCL_OPENCLMATRIX

#include <iostream>

#include "Types.h"
#include "CL/cl.h"
#include "OpenCLDevice.h"
#include "TMatrix.h"

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
//
// OpenCL Device Reference
//____________________________________________________________________________

/** TOpenCLDeviceReference
 *
 * Helper class emulating lvalue references for OpenCLDouble_t values that are
 * physically on the device. Allows for example to assign to matrix elements.
 * Note that device access through OpenCLDouble_t enforces synchronization
 * with all streams and thus qualifies as performance killer. Only used for
 * testing.
 */
class TOpenCLDeviceReference
{
private:

   TOpenCLDevice  & fDevice;
   cl_mem           fElementPointer;
   size_t           fOffset;

public:

    TOpenCLDeviceReference(cl_mem elementPointer,
                           TOpenCLDevice & device,
                           size_t offset)
    : fDevice(device), fElementPointer(elementPointer), fOffset(offset)
   {
      cl_int error;
      fDevice.HandleError(error);
   }

   operator OpenCLDouble_t()
   {
      OpenCLDouble_t buffer;
      cl_int error;
      error = clEnqueueReadBuffer(fDevice.GetQueue(), fElementPointer, CL_TRUE,
                                  fOffset * sizeof(OpenCLDouble_t),
                                  sizeof(OpenCLDouble_t), (void*) &buffer,
                                  0, nullptr, nullptr);
      fDevice.HandleError(error);
      return buffer;
   }

   void operator=(OpenCLDouble_t value)
   {
      OpenCLDouble_t buffer = value;
      cl_int error;
      error = clEnqueueWriteBuffer(fDevice.GetQueue(), fElementPointer, CL_TRUE,
                                   fOffset * sizeof(OpenCLDouble_t),
                                   sizeof(OpenCLDouble_t), (void*) &buffer,
                                   0, nullptr, nullptr);
      fDevice.HandleError(error);
   }

   void operator+=(OpenCLDouble_t value)
   {
      OpenCLDouble_t buffer;
      cl_int error;
      error = clEnqueueReadBuffer(fDevice.GetQueue(), fElementPointer, CL_TRUE,
                                  fOffset * sizeof(OpenCLDouble_t),
                                  sizeof(OpenCLDouble_t), (void*) &buffer,
                                  0, nullptr, nullptr);
      fDevice.HandleError(error);
      buffer += value;
      error = clEnqueueWriteBuffer(fDevice.GetQueue(), fElementPointer, CL_TRUE,
                                   fOffset * sizeof(OpenCLDouble_t),
                                   sizeof(OpenCLDouble_t), (void*) &buffer,
                                   0, nullptr, nullptr);
      fDevice.HandleError(error);
   }

   void operator-=(OpenCLDouble_t value)
   {
      OpenCLDouble_t buffer;
      cl_int error;
      error = clEnqueueReadBuffer(fDevice.GetQueue(), fElementPointer, CL_TRUE,
                                  fOffset * sizeof(OpenCLDouble_t),
                                  sizeof(OpenCLDouble_t), (void*) &buffer,
                                  0, nullptr, nullptr);
      fDevice.HandleError(error);
      buffer -= value;
      error = clEnqueueWriteBuffer(fDevice.GetQueue(), fElementPointer, CL_TRUE,
                                   fOffset * sizeof(OpenCLDouble_t),
                                   sizeof(OpenCLDouble_t), (void*) &buffer,
                                   0, nullptr, nullptr);
      fDevice.HandleError(error);
   }
};

class TOpenCLMatrix
{

private:

   cl_mem        fElementPointer;
   static        TOpenCLDevice fDevice;

   size_t fNRows, fNCols, fNElements;

public:

   TOpenCLMatrix(size_t nRows, size_t nCols);
   TOpenCLMatrix(const TMatrixT<OpenCLDouble_t> & A);
   operator TMatrixT<OpenCLDouble_t>() const;

   TOpenCLDeviceReference operator()(size_t i, size_t j) const
   {
      return TOpenCLDeviceReference(fElementPointer, fDevice, j * fNRows + i);
   }

   size_t             GetNrows()           const {return fNRows;}
   size_t             GetNcols()           const {return fNCols;}
   cl_mem             GetElementPointer()  const {return fElementPointer;}
   cl_command_queue   GetQueue()                 {return fDevice.GetQueue();}
   TOpenCLDevice &    GetDevice()          const {return fDevice;}

};

} // namespace DNN
} // namespace TMVA

#endif
