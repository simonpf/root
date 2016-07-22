// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Train the multi-threaded CPU implementation of DNNs on a random //
// linear mapping. In the linear case the minimization problem is  //
// convex and the gradient descent training should converge to the //
// global minimum.                                                 //
/////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

int main()
{
    testMinimization<TCpu<double, false>>();
}