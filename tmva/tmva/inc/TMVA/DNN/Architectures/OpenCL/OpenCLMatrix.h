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
#include "Buffers.h"
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

   cl::CommandQueue fQueue;
   cl::Buffer       fElementBuffer;
   size_t           fOffset;

public:

   TOpenCLDeviceReference(cl::Buffer elementBuffer,
                          cl::CommandQueue queue,
                          size_t offset)
    : fQueue(queue), fElementBuffer(elementBuffer), fOffset(offset)
   {
       // Nothing to do here.
   }

   TOpenCLDeviceReference(const TOpenCLDeviceReference  &) = default;
   TOpenCLDeviceReference(      TOpenCLDeviceReference &&) = default;
   TOpenCLDeviceReference & operator=(const TOpenCLDeviceReference  &) = default;
   TOpenCLDeviceReference & operator=(      TOpenCLDeviceReference &&) = default;

   inline operator OpenCLDouble_t();
   inline void operator =(OpenCLDouble_t value);
   inline void operator+=(OpenCLDouble_t value);
   inline void operator-=(OpenCLDouble_t value);

};

//____________________________________________________________________________
//
// OpenCL Matrix
//____________________________________________________________________________
/** Matrix class representing matrices on OpenCL devices. Each matrix has an
 *  associated TOpenCLDevice, which defaults to the default device provided by
 *  the TOpenCL architecture class. Each matrix has an associated compute which
 *  is contained in the TOpenCLDevice buffer that holds the matrix elements.
 *  This compute queue is used to ensuer consitency of the computations on
 *  OpenCL matrices.
 */
class TOpenCLMatrix
{

private:

   size_t fNRows;
   size_t fNCols;
   size_t fNElements;

   TOpenCLDeviceBuffer fElementBuffer;
   static cl::Buffer   fRandomStreams;
   static size_t       fNStreams;

public:

   TOpenCLMatrix(size_t nRows, size_t nCols);
   TOpenCLMatrix(const TMatrixT<OpenCLDouble_t> & A);
   TOpenCLMatrix(size_t nRows, size_t nCols, const TOpenCLDeviceBuffer & buffer);
   TOpenCLMatrix & operator=(const TMatrixT<OpenCLDouble_t> &);

   TOpenCLMatrix(const TOpenCLMatrix  &)             = default;
   TOpenCLMatrix(      TOpenCLMatrix &&)             = default;
   TOpenCLMatrix & operator=(const TOpenCLMatrix  &) = default;
   TOpenCLMatrix & operator=(      TOpenCLMatrix &&) = default;
   ~TOpenCLMatrix()                                  = default;

   operator TMatrixT<OpenCLDouble_t>() const;

   inline TOpenCLDeviceReference operator()(size_t i, size_t j);

   size_t     GetNrows()              const {return fNRows;}
   size_t     GetNcols()              const {return fNCols;}
   size_t     GetNoElements()         const {return fNRows * fNCols;}
   cl::Buffer GetElementBuffer()      const {return fElementBuffer;}
   cl::Buffer GetRandomStreamBuffer() const {return fRandomStreams;}

   TOpenCLDevice  & GetDevice()       const {return fElementBuffer.GetDevice();}
   cl::CommandQueue GetComputeQueue() const {return fElementBuffer.GetComputeQueue();}
   void SetComputeQueue(cl::CommandQueue queue)
   {
      fElementBuffer.SetComputeQueue(queue);
   }

private:

   inline void InitializeRandomStreams();

};

//
// Inline Functions.
//______________________________________________________________________________
TOpenCLDeviceReference::operator OpenCLDouble_t()
{
   OpenCLDouble_t buffer;
   fQueue.enqueueReadBuffer(fElementBuffer, CL_TRUE,
                            fOffset * sizeof(OpenCLDouble_t),
                            sizeof(OpenCLDouble_t),
                            (void *) &buffer);
   return buffer;
}

//______________________________________________________________________________
void TOpenCLDeviceReference::operator=(OpenCLDouble_t value)
{
   OpenCLDouble_t buffer = value;
   fQueue.enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                             fOffset * sizeof(OpenCLDouble_t),
                             sizeof(OpenCLDouble_t),
                             (void *) &buffer);
}

//______________________________________________________________________________
void TOpenCLDeviceReference::operator+=(OpenCLDouble_t value)
{
   OpenCLDouble_t buffer;
   fQueue.enqueueReadBuffer(fElementBuffer, CL_TRUE,
                            fOffset * sizeof(OpenCLDouble_t),
                            sizeof(OpenCLDouble_t),
                            (void *) &buffer);
   buffer += value;
   fQueue.enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                             fOffset * sizeof(OpenCLDouble_t),
                             sizeof(OpenCLDouble_t),
                             (void *) &buffer);
}

//______________________________________________________________________________
void TOpenCLDeviceReference::operator-=(OpenCLDouble_t value)
{
   OpenCLDouble_t buffer;
   fQueue.enqueueReadBuffer(fElementBuffer, CL_TRUE,
                            fOffset * sizeof(OpenCLDouble_t),
                            sizeof(OpenCLDouble_t),
                            (void *) &buffer);
   buffer -= value;
   fQueue.enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                             fOffset * sizeof(OpenCLDouble_t),
                             sizeof(OpenCLDouble_t),
                             (void *) &buffer);
}

//______________________________________________________________________________
TOpenCLDeviceReference TOpenCLMatrix::operator()(size_t i, size_t j)
{
   return TOpenCLDeviceReference(fElementBuffer, fElementBuffer.GetComputeQueue(),
                                 j * fNRows + i);
}

} // namespace DNN
} // namespace TMVA

#endif

