// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 05/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////
// Test the Dataloader for the OpenCL implementation. //
////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/OpenCL.h"
#include "TestDataLoader.h"

using namespace TMVA::DNN;

int main()
{

   testDataLoader<TOpenCL>();
   return 0;
}
