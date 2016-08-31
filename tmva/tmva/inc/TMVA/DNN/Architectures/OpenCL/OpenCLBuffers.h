// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh 04/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Host and device buffer classes for the OpenCL implementation of //
// deep neural networks.                                           //
/////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_OPENCL_BUFFERS
#define TMVA_DNN_ARCHITECTURES_OPENCL_BUFFERS

#include "OpenCLDevice.h"
#include "TMVA/DNN/DataLoader.h"

#include "CL/cl.hpp"
#include <iostream>
#include <memory>

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
//
// OpenCL Host Buffer Class
//____________________________________________________________________________
/** Helper class that represents buffers holding numerical values on the host.
 * Each TOpenCLHostBuffer object creates a command queue for the data transfer
 * and subsequent computations on the data.
 *
 * The host memory is allocated using the OpenCL API and should therefore be
 * pinned (if supported), which is required for asynchronous data transfers.
 *
 * If not specified otherwise the buffer will use the default device provided
 * by TOpenCL<AFloat, AType>::GetDefaultDevice() to create the command queue.
 *
 * Buffer element access is provided by [] operators.
 *
 * \tparam AFloat The floating point type used to represent real numbers.
 * \tparam AType  The type of the OpenCL device represented by an
 * EOpenCLDeviceType enum.
 */
template<typename AFloat, EOpenCLDeviceType AType>
class TOpenCLHostBuffer
{
private:

   std::shared_ptr<TOpenCLDevice<AFloat, AType>> fDevice;
   cl::CommandQueue fComputeQueue;
   cl::Buffer       fBuffer;
   AFloat *         fBufferPointer;
   size_t           fOffset;

   struct TDestructor
   {
      TDestructor() = default;
      void operator()(AFloat **hostPointer);
   } fDestructor;

public:

   /** Create a host buffer to hold \p size real numbers of the representation
    *  given by \p AFloat with a command queue on the default device.*/
   TOpenCLHostBuffer(size_t size);

   /** Create a host buffer to hold \p size real numbers of the representation
    *  given by \p AFloat with a command queue on \p device.*/
   TOpenCLHostBuffer(size_t size,
                     std::shared_ptr<TOpenCLDevice<AFloat, AType>> device);

   /** Copy a host buffer. The copied constructed buffer object will
    *  share the data with the buffer is was copied constructed *  from. */
   TOpenCLHostBuffer(const TOpenCLHostBuffer &)  = default;
   TOpenCLHostBuffer(      TOpenCLHostBuffer &&) = default;
   /** Assign a host buffer. The assigend-to buffer object will
    *  share the data  with the buffer is was copied constructed  from. */
   TOpenCLHostBuffer & operator=(const TOpenCLHostBuffer &)  = default;
   TOpenCLHostBuffer & operator=(      TOpenCLHostBuffer &&) = default;
   ~TOpenCLHostBuffer() = default;

   /** Create a sub-buffer starting at element \p offset in the buffer
    *  of size \p size. */
   TOpenCLHostBuffer GetSubBuffer(size_t offset, size_t size);
   size_t            GetOffset() const {return fOffset;}
   AFloat          * GetDataPointer()  const {return fBufferPointer + fOffset;}
   cl::CommandQueue  GetComputeQueue() const {return fComputeQueue;}

   operator AFloat * ()   const;
   operator cl::Buffer () const;

   AFloat & operator[](size_t index)       {return (fBufferPointer + fOffset)[index];}
   AFloat & operator[](size_t index) const {return (fBufferPointer + fOffset)[index];}
};

//____________________________________________________________________________
//
// OpenCL Device Buffer Class
//____________________________________________________________________________
/** Helper class for buffers holding numerical values of the given floating
 *  point type \p AFloat on OpenCL devices.
 *  An TOpenCLDeviceBuffer can be used to asynchronously copy data from a
 *  TOpenCLHostBuffer object to the device. Upon copying, the device buffer
 *  takes over the compute queue from the host buffer so that subsequent
 *  computations on the data can be performed on that queue ensuring the
 *  right order of the computations without the need to explicitly synchronize.
 *
 * \tparam AFloat The floating point type used to represent real numbers.
 * \tparam AType  The type of the OpenCL device represented by an
 * EOpenCLDeviceType enum.
 */
template<typename AFloat, EOpenCLDeviceType AType>
class TOpenCLDeviceBuffer
{
private:

   size_t                   fSize;
   size_t                   fOffset;
   cl::Buffer               fBuffer;
   mutable cl::CommandQueue fComputeQueue;
   std::shared_ptr<TOpenCLDevice<AFloat, AType>> fDevice;

   TOpenCLDeviceBuffer() = default;

public:

   /** Create a device buffer object on the default device to hold \p size
    *  numerical values of type \p AFloat */
   TOpenCLDeviceBuffer(size_t size);
   /** Create a device buffer object on \p device to hold \p size
    *  numerical values of type \p AFloat */
   TOpenCLDeviceBuffer(size_t size,
                       std::shared_ptr<TOpenCLDevice<AFloat, AType>> device);

   TOpenCLDeviceBuffer(const TOpenCLDeviceBuffer  &) = default;
   TOpenCLDeviceBuffer(      TOpenCLDeviceBuffer &&) = default;
   TOpenCLDeviceBuffer & operator=(const TOpenCLDeviceBuffer  &) = default;
   TOpenCLDeviceBuffer & operator=(      TOpenCLDeviceBuffer &&) = default;
   ~TOpenCLDeviceBuffer() = default;

   operator cl::Buffer ()       const;

   TOpenCLDeviceBuffer GetSubBuffer(size_t offset, size_t size);

   void CopyFrom(const TOpenCLHostBuffer<AFloat, AType> &) const;
   void   CopyTo(const TOpenCLHostBuffer<AFloat, AType> &) const;

   TOpenCLDevice<AFloat, AType> &  GetDevice()  const {return * fDevice;}
   cl::CommandQueue GetComputeQueue()           const {return fComputeQueue;}
   void SetComputeQueue(cl::CommandQueue queue) const {fComputeQueue = queue;}

};

} // namespace TMVA
} // namespace DNN

#endif
