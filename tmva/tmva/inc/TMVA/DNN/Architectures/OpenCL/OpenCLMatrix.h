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
#include "clRNG/lfsr113.h"

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
   cl::Buffer       fElementBuffer;
   size_t           fOffset;

public:

   TOpenCLDeviceReference(cl::Buffer elementPointer,
                          TOpenCLDevice & device,
                          size_t offset)
    : fDevice(device), fElementBuffer(elementPointer), fOffset(offset)
   {
       // Nothing to do here.
   }

   operator OpenCLDouble_t()
   {
      OpenCLDouble_t buffer;
      try {
          fDevice.GetQueue().enqueueReadBuffer(fElementBuffer, CL_TRUE,
                                               fOffset * sizeof(OpenCLDouble_t),
                                               sizeof(OpenCLDouble_t),
                                               (void *) &buffer);
      } catch (cl::Error error) {
          fDevice.HandleError(error.err());
      }
      return buffer;
   }

   void operator=(OpenCLDouble_t value)
   {
      OpenCLDouble_t buffer = value;
      try {
          fDevice.GetQueue().enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                                                fOffset * sizeof(OpenCLDouble_t),
                                                sizeof(OpenCLDouble_t),
                                                (void *) &buffer);
      } catch (cl::Error error) {
          fDevice.HandleError(error.err());
      }
   }

   void operator+=(OpenCLDouble_t value)
   {
      OpenCLDouble_t buffer;
      try {
          fDevice.GetQueue().enqueueReadBuffer(fElementBuffer, CL_TRUE,
                                               fOffset * sizeof(OpenCLDouble_t),
                                               sizeof(OpenCLDouble_t),
                                               (void *) &buffer);
          buffer += value;
          fDevice.GetQueue().enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                                                fOffset * sizeof(OpenCLDouble_t),
                                                sizeof(OpenCLDouble_t),
                                                (void *) &buffer);
      } catch (cl::Error error) {
          fDevice.HandleError(error.err());
      }
   }

   void operator-=(OpenCLDouble_t value)
   {
      OpenCLDouble_t buffer;
      try {
          fDevice.GetQueue().enqueueReadBuffer(fElementBuffer, CL_TRUE,
                                               fOffset * sizeof(OpenCLDouble_t),
                                               sizeof(OpenCLDouble_t),
                                               (void *) &buffer);
          buffer -= value;
          fDevice.GetQueue().enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                                                fOffset * sizeof(OpenCLDouble_t),
                                                sizeof(OpenCLDouble_t),
                                                (void *) &buffer);
      } catch (cl::Error error) {
          fDevice.HandleError(error.err());
      }
   }
};

class TOpenCLMatrix
{

private:

          cl::Buffer    fElementBuffer;
   static TOpenCLDevice fDevice;
   static cl::Buffer    fRandomStreams;
   static size_t        fNStreams;

   size_t fNRows, fNCols, fNElements;

public:

   TOpenCLMatrix(size_t nRows, size_t nCols);
   TOpenCLMatrix(const TMatrixT<OpenCLDouble_t> & A);

   TOpenCLMatrix(const TOpenCLMatrix &)             = default;
   TOpenCLMatrix & operator=(const TOpenCLMatrix &) = delete;
   ~TOpenCLMatrix()                                 = default;

   operator TMatrixT<OpenCLDouble_t>() const;
   TOpenCLMatrix & operator=(const TMatrixT<OpenCLDouble_t> &);

   TOpenCLDeviceReference operator()(size_t i, size_t j) const
   {
      return TOpenCLDeviceReference(fElementBuffer, fDevice, j * fNRows + i);
   }

   size_t               GetNrows()               const {return fNRows;}
   size_t               GetNcols()               const {return fNCols;}
   size_t               GetNoElements()          const {return fNRows * fNCols;}
   cl::Buffer           GetElementBuffer()       const {return fElementBuffer;}
   cl::Buffer           GetRandomStreamBuffer()  const {return fRandomStreams;}
   cl::CommandQueue     GetQueue()               const {return fDevice.GetQueue();}
   TOpenCLDevice &      GetDevice()              const {return fDevice;}

private:

   inline void InitializeRandomStreams();

};

} // namespace DNN
} // namespace TMVA

#endif
