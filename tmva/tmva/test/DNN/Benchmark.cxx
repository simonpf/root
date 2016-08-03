// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////
// Let DNN learn simple linear mapping. Used for profiling. //
//////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

using Architecture = TCuda<true>;

int main()
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TNet<Architecture>;

   Matrix_t XTrain(10000,20), YTrain(10000,20), XTest(1000,20), YTest(1000,20),
            W(20, 1);

   // Generate test data.
   randomMatrix(W);
   randomMatrix(XTrain);
   randomMatrix(XTest);
   YTrain.MultT(XTrain, W);
   YTest.MultT(XTest, W);

   Net_t net(1000, 20, ELossFunction::MEANSQUAREDERROR);
   net.AddLayer(100, EActivationFunction::IDENTITY);
   net.AddLayer(100, EActivationFunction::IDENTITY);
   net.AddLayer(100, EActivationFunction::IDENTITY);
   net.AddLayer(100, EActivationFunction::IDENTITY);
   net.AddLayer(100, EActivationFunction::IDENTITY);
   net.Initialize(EInitialization::GAUSS);

   TGradientDescent<Architecture> minimizer(0.01, 20, 20);
   MatrixInput_t trainingData(XTrain, YTrain);
   MatrixInput_t testData(XTest, YTest);

   minimizer.Train(trainingData, 10000, testData, 20, net);

   return 0;
}
