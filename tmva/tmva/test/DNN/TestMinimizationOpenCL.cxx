// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 07/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Test the task-based minimization for the OpenCL architecture. //
///////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/OpenCL.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

int main ()
{
    testMinimization<TOpenCL>();
}

