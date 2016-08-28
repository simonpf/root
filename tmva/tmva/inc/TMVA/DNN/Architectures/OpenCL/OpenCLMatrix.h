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

#include "CL/cl.h"
#include "OpenCLDevice.h"
#include "OpenCLBuffers.h"
#include "TMatrix.h"
#include "Rtypes.h"
#include "clRNG/lfsr113.h"

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
//
// OpenCL Device Reference
//____________________________________________________________________________
/** Helper class emulating lvalue references for scalar numeric
 *  variables that are physically on the device. Allows for example to
 *  assign to matrix elements.  Note that device access through an
 *  TOpenCLDeviceReference object enforces synchronization with all
 *  streams and thus qualifies as performance killer. Only used for
 *  testing.
 *
 *  \tparam The floating point type used for the representation of
 *  matrix elements.
 */
template<typename AFloat>
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

   inline operator AFloat();
   inline void operator =(AFloat value);
   inline void operator+=(AFloat value);
   inline void operator-=(AFloat value);

};

//____________________________________________________________________________
//
// OpenCL Matrix
//____________________________________________________________________________
/** Matrix class representing matrices on OpenCL devices. Each matrix has an
 *  associated TOpenCLDevice, which defaults to the default device
 *  provided by the TOpenCL architecture class. Each matrix has an
 *  associated compute stream associated with the TOpenCLDevice buffer
 *  that holds the matrix elements.  This compute queue can be used to
 *  synchronized dependent computations on OpenCL matrices.
 *
 *  The TOpenCLMatrix class has a static, associated buffer of clRNG random
 *  stream that provides as many random streams as the number of elements of
 *  the largest, current TOpenCLMatrix instance.
 *
 * \tparam AFloat The floating point type used to represent matrix elements.
 */
template <typename AFloat, EOpenCLDeviceType AType>
class TOpenCLMatrix
{

private:

   size_t fNRows;
   size_t fNCols;
   size_t fNElements;

   static cl::Buffer   fRandomStreams;
   static size_t       fNStreams;
   TOpenCLDeviceBuffer<AFloat, AType> fElementBuffer;

public:

   /** Construct a new \p nRows x \p nCols matrix on the device. Allocates the
    *  necessary memory. The compute queue defaults to the default compute queue
    *  of the TOpenCL classes default device. */
   TOpenCLMatrix(size_t nRows, size_t nCols);
   /** Copy a TMatrixT<Double_t> object from the host. Allocates all necessary
    *  memory and sets the compute queue to the default computed queue of the
    *  TOpenCL architecture class.*/
   TOpenCLMatrix(const TMatrixT<Double_t> & A);
   /** Construct a TOpenCLMatrix from an existing device buffer \p buffer of
    *  size \p nRows x \p nCols. No memory is allocated, no data
    * copied and the compute queue is inherited from the device
    * buffer. */
   TOpenCLMatrix(const TOpenCLDeviceBuffer<AFloat, AType> & buffer,
                 size_t nRows,
                 size_t nCols);
   /** Copy elements from an TMatrixT<Double_t> matrix to a matrix on the
    *  device. If the sizes are not compatible behaviout is undefined. */
   TOpenCLMatrix & operator=(const TMatrixT<Double_t> &);

   /** Performs a shallow copy. The constructed matrix shares the element buffer
    *  with the matrix it has been constructed from. */
   TOpenCLMatrix(const TOpenCLMatrix  &)             = default;
   TOpenCLMatrix(      TOpenCLMatrix &&)             = default;
   /** Performs a shallow copy. The assigned-to matrix shares the element buffer
    *  with the matrix that has been assigned from. */
   TOpenCLMatrix & operator=(const TOpenCLMatrix  &) = default;
   TOpenCLMatrix & operator=(      TOpenCLMatrix &&) = default;
   ~TOpenCLMatrix()                                  = default;

   /** Copy matrix data to a TMatrixT<Double_t> matrix on
    *  the host. The data transfer is synchronous.*/
   operator TMatrixT<Double_t>() const;

   /** Access the element in row \p i and column \p j using via a
    *  TOpenCLDeviceReference object. */
   inline TOpenCLDeviceReference<AFloat> operator()(size_t i, size_t j);
   inline AFloat                         operator()(size_t i, size_t j) const;

   size_t     GetNrows()              const {return fNRows;}
   size_t     GetNcols()              const {return fNCols;}
   size_t     GetNoElements()         const {return fNRows * fNCols;}
   cl::Buffer GetElementBuffer()      const {return fElementBuffer;}
   cl::Buffer GetRandomStreamBuffer() const {return fRandomStreams;}
   cl::CommandQueue GetComputeQueue() const {return fElementBuffer.GetComputeQueue();}

   TOpenCLDevice<AFloat, AType> & GetDevice() const
   {
      return fElementBuffer.GetDevice();
   }

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
template<typename AFloat>
TOpenCLDeviceReference<AFloat>::operator AFloat()
{
   AFloat buffer;
   fQueue.enqueueReadBuffer(fElementBuffer, CL_TRUE,
                            fOffset * sizeof(AFloat),
                            sizeof(AFloat),
                            (void *) &buffer);
   return buffer;
}

//______________________________________________________________________________
template<typename AFloat>
void TOpenCLDeviceReference<AFloat>::operator=(AFloat value)
{
   AFloat buffer = value;
   fQueue.enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                             fOffset * sizeof(AFloat),
                             sizeof(AFloat),
                             (void *) &buffer);
}

//______________________________________________________________________________
template<typename AFloat>
void TOpenCLDeviceReference<AFloat>::operator+=(AFloat value)
{
   AFloat buffer;
   fQueue.enqueueReadBuffer(fElementBuffer, CL_TRUE,
                            fOffset * sizeof(AFloat),
                            sizeof(AFloat),
                            (void *) &buffer);
   buffer += value;
   fQueue.enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                             fOffset * sizeof(AFloat),
                             sizeof(AFloat),
                             (void *) &buffer);
}

//______________________________________________________________________________
template<typename AFloat>
void TOpenCLDeviceReference<AFloat>::operator-=(AFloat value)
{
   AFloat buffer;
   fQueue.enqueueReadBuffer(fElementBuffer, CL_TRUE,
                            fOffset * sizeof(AFloat),
                            sizeof(AFloat),
                            (void *) &buffer);
   buffer -= value;
   fQueue.enqueueWriteBuffer(fElementBuffer, CL_TRUE,
                             fOffset * sizeof(AFloat),
                             sizeof(AFloat),
                             (void *) &buffer);
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
auto TOpenCLMatrix<AFloat, AType>::operator()(size_t i, size_t j)
    -> TOpenCLDeviceReference<AFloat>
{
   return TOpenCLDeviceReference<AFloat>(fElementBuffer,
                                           fElementBuffer.GetComputeQueue(),
                                           j * fNRows + i);
}

//______________________________________________________________________________
template<typename AFloat, EOpenCLDeviceType AType>
AFloat TOpenCLMatrix<AFloat, AType>::operator()(size_t i, size_t j) const
{
   return TOpenCLDeviceReference<AFloat>(fElementBuffer,
                                           fElementBuffer.GetComputeQueue(),
                                           j * fNRows + i);
}

} // namespace DNN
} // namespace TMVA

#endif

