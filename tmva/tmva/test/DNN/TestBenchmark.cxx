
#include "TMatrix.h"
#include "Utility.h"
#include "TMVA/DNN/Net.h"
#include "TMVA/DNN/Architectures/Cuda.h"

using namespace TMVA::DNN;

int main()
{
   TMatrixT<Double_t> X(4000,20), Y(4000,20);
   randomMatrix(X);
   randomMatrix(Y);
   MatrixInput_t Data(X, Y);

   using Architecture = TCuda<true>;
   using Net = TNet<Architecture>;
   typename Architecture::DataLoader_t<MatrixInput_t> loader(Data, 4000, 20, 20, 20);

   Net net(20, 20, ELossFunction::CROSSENTROPY);
   net.AddLayer(200, EActivationFunction::IDENTITY);
   net.AddLayer(200, EActivationFunction::SIGMOID);
   net.AddLayer(200, EActivationFunction::TANH);
   net.AddLayer(200, EActivationFunction::IDENTITY);
   net.AddLayer(20, EActivationFunction::IDENTITY);

   for (auto b : loader) {
      auto && input = b.GetInput();
      net.Loss(input, b.GetOutput());
      net.Backward(input, b.GetOutput());
   }

   Architecture::GetTimings().Print();
}
