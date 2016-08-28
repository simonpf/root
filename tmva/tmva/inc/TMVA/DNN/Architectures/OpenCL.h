// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Definition of the TOpenCL architecture, which provides an    //
// implementation of the low-level functionality for neural     //
// networks for OpenCL computing architectures.                 //
//////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_OPENCL
#define TMVA_DNN_ARCHITECTURES_OPENCL

#include <utility>
#include <memory>

#include "Rtypes.h"
#include "OpenCL/OpenCLDevice.h"
#include "OpenCL/OpenCLMatrix.h"
#include "OpenCL/OpenCLBuffers.h"
#include "CL/cl.h"

namespace TMVA
{
namespace DNN
{

/** The TOpenCL architecture class.
 *
 * Low-level interface class for OpenCL computing architectures. Contains as
 * public types the declaration of the scalar, matrix and data loader types
 * for this architecture as well as the remaining functions in the low-level
 * interface in the form of static members.
 */
template<typename AFloat, EOpenCLDeviceType AType = EOpenCLDeviceType::kGpu>
class TOpenCL
{

private:

   /** Provides an OpenCL default device and context to be used if no unified
    *  device handling is available. Creates the device only if requested. */
   static std::shared_ptr<TOpenCLDevice<AFloat, AType>> fDefaultDevice;

public:

   using Scalar_t       = AFloat;
   using Matrix_t       = TOpenCLMatrix<AFloat, AType>;
   using DeviceBuffer_t = TOpenCLDeviceBuffer<AFloat, AType>;
   using HostBuffer_t   = TOpenCLHostBuffer<AFloat, AType>;

   /** Return a reference to the default device or create it if it has not
    *  yet been created. */
   static std::shared_ptr<TOpenCLDevice<AFloat, AType>> GetDefaultDevice();

   //____________________________________________________________________________
   //
   // Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
   ///@{
   /** Matrix-multiply \p input with the transpose of \pweights and
    *  write the results into \p output. */
   static void MultiplyTranspose(TOpenCLMatrix<AFloat, AType> &output,
                                 const TOpenCLMatrix<AFloat, AType> &input,
                                 const TOpenCLMatrix<AFloat, AType> &weights);
   /** Add the vectors biases row-wise to the matrix output */
   static void AddRowWise(TOpenCLMatrix<AFloat, AType> &output,
                          const TOpenCLMatrix<AFloat, AType> &biases);
   ///@}

   /** @name Backward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
   ///@{
   /** Perform the complete backward propagation step. If the provided
    *  \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void Backward(TOpenCLMatrix<AFloat, AType> & activationGradientsBackward,
                        TOpenCLMatrix<AFloat, AType> & weightGradients,
                        TOpenCLMatrix<AFloat, AType> & biasGradients,
                        TOpenCLMatrix<AFloat, AType> & df,
                        const TOpenCLMatrix<AFloat, AType> & activationGradients,
                        const TOpenCLMatrix<AFloat, AType> & weights,
                        const TOpenCLMatrix<AFloat, AType> & activationBackward);
   /** Adds a the elements in matrix B scaled by c to the elements in
    *  the matrix A. This is required for the weight update in the gradient
    *  descent step.*/
   static void ScaleAdd(TOpenCLMatrix<AFloat, AType> & A,
                        const TOpenCLMatrix<AFloat, AType> & B,
                        Scalar_t beta = 1.0);

   static void Copy(TOpenCLMatrix<AFloat, AType> & B,
                    const TOpenCLMatrix<AFloat, AType> & A);
   ///@}

   //____________________________________________________________________________
   //
   // Activation Functions
   //____________________________________________________________________________

   /** @name Activation Functions
    * For each activation function, the low-level interface contains two routines.
    * One that applies the acitvation function to a matrix and one that evaluate
    * the derivatives of the activation function at the elements of a given matrix
    * and writes the results into the result matrix.
    */
   ///@{
   static void Identity(TOpenCLMatrix<AFloat, AType> & B);
   static void IdentityDerivative(TOpenCLMatrix<AFloat, AType> & B,
                                  const TOpenCLMatrix<AFloat, AType> & A);

   static void Relu(TOpenCLMatrix<AFloat, AType> & B);
   static void ReluDerivative(TOpenCLMatrix<AFloat, AType> & B,
                              const TOpenCLMatrix<AFloat, AType> & A);

   static void Sigmoid(TOpenCLMatrix<AFloat, AType> & B);
   static void SigmoidDerivative(TOpenCLMatrix<AFloat, AType> & B,
                                 const TOpenCLMatrix<AFloat, AType> & A);

   static void Tanh(TOpenCLMatrix<AFloat, AType> & B);
   static void TanhDerivative(TOpenCLMatrix<AFloat, AType> & B,
                              const TOpenCLMatrix<AFloat, AType> & A);

   static void SymmetricRelu(TOpenCLMatrix<AFloat, AType> & B);
   static void SymmetricReluDerivative(TOpenCLMatrix<AFloat, AType> & B,
                                       const TOpenCLMatrix<AFloat, AType> & A);

   static void SoftSign(TOpenCLMatrix<AFloat, AType> & B);
   static void SoftSignDerivative(TOpenCLMatrix<AFloat, AType> & B,
                                  const TOpenCLMatrix<AFloat, AType> & A);

   static void Gauss(TOpenCLMatrix<AFloat, AType> & B);
   static void GaussDerivative(TOpenCLMatrix<AFloat, AType> & B,
                               const TOpenCLMatrix<AFloat, AType> & A);
   ///@}

   //____________________________________________________________________________
   //
   // Loss Functions
   //____________________________________________________________________________

   /** @name Loss Functions
    * Loss functions compute a scalar value given the \p output of the network
    * for a given training input and the expected network prediction \p Y that
    * quantifies the quality of the prediction. For each function also a routing
    * that computes the gradients (suffixed by Gradients) must be provided for
    * the starting of the backpropagation algorithm.
    */
   ///@{

   static AFloat MeanSquaredError(const TOpenCLMatrix<AFloat, AType> & Y,
                                  const TOpenCLMatrix<AFloat, AType> & output);
   static void MeanSquaredErrorGradients(      TOpenCLMatrix<AFloat, AType> & dY,
                                         const TOpenCLMatrix<AFloat, AType> & Y,
                                         const TOpenCLMatrix<AFloat, AType> & output);

    /** Sigmoid transformation is implicitly applied, thus \p output should
     *  hold the linear activations of the last layer in the net. */
   static AFloat CrossEntropy(const TOpenCLMatrix<AFloat, AType> & Y,
                              const TOpenCLMatrix<AFloat, AType> & output);
   static void CrossEntropyGradients(      TOpenCLMatrix<AFloat, AType> & dY,
                                     const TOpenCLMatrix<AFloat, AType> & Y,
                                     const TOpenCLMatrix<AFloat, AType> & output);
   ///@}

   //____________________________________________________________________________
   //
   // Output Functions
   //____________________________________________________________________________

   /** @name Output Functions
    * Output functions transform the activations \p output of the
    * output layer in the network to a valid prediction \p YHat for
    * the desired usage of the network, e.g.  the identity function
    * for regression or the sigmoid transformation for two-class
    * classification.
    */
   ///@{
   static void Sigmoid(TOpenCLMatrix<AFloat, AType> &YHat,
                       const TOpenCLMatrix<AFloat, AType> & );
   ///@}

   //____________________________________________________________________________
   //
   // Regularization
   //____________________________________________________________________________

   /** @name Regularization
    * For each regularization type two functions are required, one named
    * <tt><Type>Regularization</tt> that evaluates the corresponding
    * regularization functional for a given weight matrix and the
    * <tt>Add<Type>RegularizationGradients</tt>, that adds the regularization
    * component in the gradients to the provided matrix.
    */
   ///@{

   static AFloat L1Regularization(const TOpenCLMatrix<AFloat, AType> & W);
   static void AddL1RegularizationGradients(TOpenCLMatrix<AFloat, AType> & A,
                                            const TOpenCLMatrix<AFloat, AType> & W,
                                            AFloat weightDecay);

   static AFloat L2Regularization(const TOpenCLMatrix<AFloat, AType> & W);
   static void AddL2RegularizationGradients(TOpenCLMatrix<AFloat, AType> & A,
                                            const TOpenCLMatrix<AFloat, AType> & W,
                                            AFloat weightDecay);
   ///@}

   //____________________________________________________________________________
   //
   // Initialization
   //____________________________________________________________________________

   /** @name Initialization
    * For each initialization method, one function in the low-level interface
    * is provided. The naming scheme is <p>Initialize<Type></p> for a given
    * initialization method Type.
    */
   ///@{

   static void InitializeGauss(TOpenCLMatrix<AFloat, AType> & A);
   static void InitializeUniform(TOpenCLMatrix<AFloat, AType> & A);
   static void InitializeIdentity(TOpenCLMatrix<AFloat, AType> & A);
   static void InitializeZero(TOpenCLMatrix<AFloat, AType> & A);

   ///@}

   //____________________________________________________________________________
   //
   // Dropout
   //____________________________________________________________________________

   /** @name Dropout
    */
   ///@{

   /** Apply dropout with activation probability \p p to the given
    *  matrix \p A and scale the result by reciprocal of \p p. */
   static void Dropout(TOpenCLMatrix<AFloat, AType> & A, AFloat p);

   ///@}

   //____________________________________________________________________________
   //
   // Additional Arithmetic Functions
   //____________________________________________________________________________

   /** @name Additional Arithmetic Functions
    *
    * Additional arithmetic on CUDA matrices  used to implement the low-level
    * interface.
    */
   ///@{

   /** Standard multiplication of two matrices \p A and \p B with the result being
    *  written into C.
    */
   static void Multiply(TOpenCLMatrix<AFloat, AType> &C,
                        const TOpenCLMatrix<AFloat, AType> &A,
                        const TOpenCLMatrix<AFloat, AType> &B);
   /** Matrix multiplication of two matrices \p A and \p B^T (transposed) with the
    *  result being written into C.
    */
   static void TransposeMultiply(TOpenCLMatrix<AFloat, AType> &output,
                                 const TOpenCLMatrix<AFloat, AType> &input,
                                 const TOpenCLMatrix<AFloat, AType> &Weights);
   /** In-place Hadamard (element-wise) product of matrices \p A and \p B
    *  with the result being written into \p A.
    */
   static void Hadamard(TOpenCLMatrix<AFloat, AType> &A,
                        const TOpenCLMatrix<AFloat, AType> &B);

   /** Sum columns of (m x n) matrixx \p A and write the results into the first
    * m elements in \p A.
    */
   static void SumColumns(TOpenCLMatrix<AFloat, AType> &B,
                          const TOpenCLMatrix<AFloat, AType> &A);

   /** Compute the sum of all elements in \p A */
   static AFloat Sum(const TOpenCLMatrix<AFloat, AType> &A);
};

} // namespace DNN
} // namespace TMVA

#endif
