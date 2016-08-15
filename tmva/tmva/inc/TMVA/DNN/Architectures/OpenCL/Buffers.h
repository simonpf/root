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

#include "Types.h"
#include "TMVA/DNN/DataLoader.h"

#include "CL/cl.hpp"
#include <iostream>
#include <memory>

namespace TMVA {
namespace DNN  {

class TOpenCLHostBuffer
{
private:

   std::shared_ptr<TOpenCLDevice> fDevice;
   cl::CommandQueue               fComputeQueue;
   cl::Buffer                     fBuffer;
   OpenCLDouble_t *               fBufferPointer;
   size_t                         fOffset;

   struct TDestructor
   {
       TDestructor() = default;
       void operator()(OpenCLDouble_t **hostPointer);
   } fDestructor;

public:

   TOpenCLHostBuffer(size_t size);
   TOpenCLHostBuffer(size_t size, std::shared_ptr<TOpenCLDevice> device);
   TOpenCLHostBuffer(const TOpenCLHostBuffer &)  = default;
   TOpenCLHostBuffer(      TOpenCLHostBuffer &&) = default;
   TOpenCLHostBuffer & operator=(const TOpenCLHostBuffer &)  = default;
   TOpenCLHostBuffer & operator=(      TOpenCLHostBuffer &&) = default;
   ~TOpenCLHostBuffer() = default;

   TOpenCLHostBuffer GetSubBuffer(size_t offset, size_t size);
   size_t            GetOffset() const {return fOffset;}
   OpenCLDouble_t *  GetDataPointer()  const {return fBufferPointer + fOffset;}
   cl::CommandQueue  GetComputeQueue() const {return fComputeQueue;}

   operator OpenCLDouble_t * () const;
   operator cl::Buffer ()       const;

   OpenCLDouble_t & operator[](size_t index)
   {
      return (fBufferPointer + fOffset)[index];
   }
   OpenCLDouble_t   operator[](size_t index)   const
   {
      return (fBufferPointer + fOffset)[index];
   }
};

class TOpenCLDeviceBuffer
{
private:

   size_t                         fSize;
   size_t                         fOffset;
   cl::Buffer                     fBuffer;
   std::shared_ptr<TOpenCLDevice> fDevice;
   mutable cl::CommandQueue       fComputeQueue;

   TOpenCLDeviceBuffer() = default;

public:

   TOpenCLDeviceBuffer(size_t size);
   TOpenCLDeviceBuffer(size_t size, std::shared_ptr<TOpenCLDevice> fDevice);
   TOpenCLDeviceBuffer(const TOpenCLDeviceBuffer  &) = default;
   TOpenCLDeviceBuffer(      TOpenCLDeviceBuffer &&) = default;
   TOpenCLDeviceBuffer & operator=(const TOpenCLDeviceBuffer  &) = default;
   TOpenCLDeviceBuffer & operator=(      TOpenCLDeviceBuffer &&) = default;
   ~TOpenCLDeviceBuffer() = default;

   operator cl::Buffer ()       const;

   TOpenCLDeviceBuffer GetSubBuffer(size_t offset, size_t size);

   void CopyFrom(const TOpenCLHostBuffer &) const;
   void   CopyTo(const TOpenCLHostBuffer &) const;

   TOpenCLDevice &  GetDevice()       const {return * fDevice;}
   cl::CommandQueue GetComputeQueue() const {return fComputeQueue;}
   void SetComputeQueue(cl::CommandQueue queue) {fComputeQueue = queue;}

};

} // namespace TMVA
} // namespace DNN

#endif
