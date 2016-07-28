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

   size_t             GetNrows()           const {return fNRows;}
   size_t             GetNcols()           const {return fNCols;}
   cl_mem             GetElementPointer()  const {return fElementPointer;}
   cl_command_queue   GetQueue()                 {return fDevice.GetQueue();}
   TOpenCLDevice &    GetDevice()          const {return fDevice;}

};

} // namespace DNN
} // namespace TMVA

#endif
