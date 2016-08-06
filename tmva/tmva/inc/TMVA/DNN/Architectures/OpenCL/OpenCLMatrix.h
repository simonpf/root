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

   const TOpenCLDevice & fDevice;
   cl::Buffer fElementBuffer;
   size_t     fOffset;

public:

   TOpenCLDeviceReference(cl::Buffer elementPointer,
                          const TOpenCLDevice & device,
                          size_t offset)
    : fDevice(device), fElementBuffer(elementPointer), fOffset(offset)
   {
       // Nothing to do here.
   }

   operator OpenCLDouble_t()
   {
      OpenCLDouble_t buffer;
      try {
          fDevice.GetQueue(0).enqueueReadBuffer(fElementBuffer, CL_TRUE,
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
          fDevice.GetQueue(0).enqueueWriteBuffer(fElementBuffer, CL_TRUE,
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
          fDevice.GetQueue(0).enqueueReadBuffer(fElementBuffer, CL_TRUE,
                                                fOffset * sizeof(OpenCLDouble_t),
                                                sizeof(OpenCLDouble_t),
                                                (void *) &buffer);
          buffer += value;
          fDevice.GetQueue(0).enqueueWriteBuffer(fElementBuffer, CL_TRUE,
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
          fDevice.GetQueue(0).enqueueReadBuffer(fElementBuffer, CL_TRUE,
                                                fOffset * sizeof(OpenCLDouble_t),
                                                sizeof(OpenCLDouble_t),
                                                (void *) &buffer);
          buffer -= value;
          fDevice.GetQueue(0).enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                                                 fOffset * sizeof(OpenCLDouble_t),
                                                 sizeof(OpenCLDouble_t),
                                                 (void *) &buffer);
      } catch (cl::Error error) {
          fDevice.HandleError(error.err());
      }
   }
};

static TOpenCLDevice DefaultDevice{};

class TOpenCLMatrix
{

private:

    using DeviceBuffer_t = TOpenCLDevice::TOpenCLDeviceBuffer;

          cl::Buffer    fElementBuffer;
   const TOpenCLDevice & fDevice;
   static cl::Buffer    fRandomStreams;
   static size_t        fNStreams;

   size_t fNRows;
   size_t fNCols;
   size_t fNElements;
   size_t fComputeStreamIndex;

public:

   TOpenCLMatrix(size_t nRows,
                 size_t nCols,
                 const TOpenCLDevice & device = DefaultDevice,
                 size_t computeStreamIndex = 0);
   TOpenCLMatrix(const TMatrixT<OpenCLDouble_t> & A,
                 const TOpenCLDevice & device = DefaultDevice,
                 size_t computeStreamIndex = 0);
   TOpenCLMatrix(size_t nRows,
                 size_t nCols,
                 const  DeviceBuffer_t & buffer,
                 size_t computeStreamIndex);

   TOpenCLMatrix(const TOpenCLMatrix &)             = default;
   TOpenCLMatrix & operator=(const TOpenCLMatrix &) = delete;
   ~TOpenCLMatrix()                                 = default;

   operator TMatrixT<OpenCLDouble_t>() const;
   TOpenCLMatrix & operator=(const TMatrixT<OpenCLDouble_t> &);

   TOpenCLDeviceReference operator()(size_t i, size_t j) const
   {
      return TOpenCLDeviceReference(fElementBuffer, fDevice, j * fNRows + i);
   }

   size_t     GetNrows()              const {return fNRows;}
   size_t     GetNcols()              const {return fNCols;}
   size_t     GetNoElements()         const {return fNRows * fNCols;}
   cl::Buffer GetElementBuffer()      const {return fElementBuffer;}
   cl::Buffer GetRandomStreamBuffer() const {return fRandomStreams;}
   size_t     GetComputeStreamIndex() const {return fComputeStreamIndex;}

   const TOpenCLDevice & GetDevice() const {return fDevice;}
   cl::CommandQueue      GetQueue()  const
   {
       return fDevice.GetQueue(fComputeStreamIndex);
   }

   void SetComputeStreamIndex(size_t index) {fComputeStreamIndex = index;}


private:

   inline void InitializeRandomStreams();

};

} // namespace DNN
} // namespace TMVA

#endif
