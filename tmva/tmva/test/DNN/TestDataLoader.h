// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 12/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////
// Generic test for DataLoader implementations. //
//////////////////////////////////////////////////

#include "TMVA/DNN/DeviceDataLoader.h"
#include "TMatrix.h"
#include "Utility.h"
#include "tbb/tbb.h"

namespace TMVA
{
namespace DNN
{

template <typename Architecture_t>
auto testDataLoader()
    -> typename Architecture_t::Scalar_t
{

   using Real_t       = typename Architecture_t::Scalar_t;
   using Device_t     = typename Architecture_t::Device_t;
   using DataLoader_t = TDeviceDataLoader<Architecture_t, MatrixInput_t>;

   size_t nSamples  = 1000000;
   size_t batchSize = 10000;
   size_t nFeatures = 100;

   TMatrixT<Double_t> X(nSamples, nFeatures), Y(nSamples, nFeatures);
   randomMatrix(X);
   randomMatrix(Y);
   MatrixInput_t input(X, Y);

   Device_t     device(4);
   DataLoader_t loader(input, nSamples, batchSize, nFeatures, nFeatures, 4, 4, device);

   Real_t msq = 0.0;
   for (size_t i = 0; i < nSamples / batchSize; i++) {
      auto batch = loader.GetBatch();
      msq += Architecture_t::MeanSquaredError(batch.GetInput(), batch.GetOutput());
   }

   X -= Y;
   Real_t msqRef = X.E2Norm();
   msq    *= batchSize * nFeatures;
   std::cout << "Maximum relative error: " << (msq - msqRef) / msqRef << std::endl;

   return 0.0;
}

} // namespace DNN
} // namespace TMVA
