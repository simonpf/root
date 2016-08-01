// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 05/07/16

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

#include "OpenCL/Types.h"
#include "OpenCL/OpenCLMatrix.h"
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
class TOpenCL
{

public:

    using Scalar_t   = OpenCLDouble_t;
    using Matrix_t   = TOpenCLMatrix;

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
   static void MultiplyTranspose(TOpenCLMatrix &output,
                                 const TOpenCLMatrix &input,
                                 const TOpenCLMatrix &weights);
   /** Add the vectors biases row-wise to the matrix output */
   static void AddRowWise(TOpenCLMatrix &output,
                          const TOpenCLMatrix &biases);
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
   static void Backward(TOpenCLMatrix & activationGradientsBackward,
                        TOpenCLMatrix & weightGradients,
                        TOpenCLMatrix & biasGradients,
                        TOpenCLMatrix & df,
                        const TOpenCLMatrix & activationGradients,
                        const TOpenCLMatrix & weights,
                        const TOpenCLMatrix & activationBackward);
   /** Adds a the elements in matrix B scaled by c to the elements in
    *  the matrix A. This is required for the weight update in the gradient
    *  descent step.*/
   static void ScaleAdd(TOpenCLMatrix & A,
                        const TOpenCLMatrix & B,
                        Scalar_t beta = 1.0);
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
   static void Identity(TOpenCLMatrix & B);
   static void IdentityDerivative(TOpenCLMatrix & B);

   static void Relu(TOpenCLMatrix & B);
   static void ReluDerivative(TOpenCLMatrix & B,
                              const TOpenCLMatrix & A);

   static void Sigmoid(TOpenCLMatrix & B);
   static void SigmoidDerivative(TOpenCLMatrix & B,
                                 const TOpenCLMatrix & A);

   static void Tanh(TOpenCLMatrix & B);
   static void TanhDerivative(TOpenCLMatrix & B,
                              const TOpenCLMatrix & A);

   static void SymmetricRelu(TOpenCLMatrix & B);
   static void SymmetricReluDerivative(TOpenCLMatrix & B,
                                       const TOpenCLMatrix & A);

   static void SoftSign(TOpenCLMatrix & B);
   static void SoftSignDerivative(TOpenCLMatrix & B,
                                  const TOpenCLMatrix & A);

   static void Gauss(TOpenCLMatrix & B);
   static void GaussDerivative(TOpenCLMatrix & B,
                               const TOpenCLMatrix & A);
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

   static OpenCLDouble_t MeanSquaredError(const TOpenCLMatrix &Y,
                                        const TOpenCLMatrix & output);
   static void MeanSquaredErrorGradients(TOpenCLMatrix & dY,
                                         const TOpenCLMatrix & Y,
                                         const TOpenCLMatrix & output);

    /** Sigmoid transformation is implicitly applied, thus \p output should
     *  hold the linear activations of the last layer in the net. */
   static OpenCLDouble_t CrossEntropy(const TOpenCLMatrix &Y,
                              const TOpenCLMatrix &output);

   static void CrossEntropyGradients(TOpenCLMatrix & dY,
                                     const TOpenCLMatrix & Y,
                                     const TOpenCLMatrix & output);
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
   static void Sigmoid(TOpenCLMatrix &YHat,
                        const TOpenCLMatrix & );
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

   static OpenCLDouble_t L1Regularization(const TOpenCLMatrix & W);
   static void AddL1RegularizationGradients(TOpenCLMatrix & A,
                                            const TOpenCLMatrix & W,
                                            OpenCLDouble_t weightDecay);

   static OpenCLDouble_t L2Regularization(const TOpenCLMatrix & W);
   static void AddL2RegularizationGradients(TOpenCLMatrix & A,
                                            const TOpenCLMatrix & W,
                                            OpenCLDouble_t weightDecay);
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

   static void InitializeGauss(TOpenCLMatrix & A);
   static void InitializeUniform(TOpenCLMatrix & A);
   static void InitializeIdentity(TOpenCLMatrix & A);
   static void InitializeZero(TOpenCLMatrix & A);

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
   static void Dropout(TOpenCLMatrix & A, OpenCLDouble_t p);

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
   static void Multiply(TOpenCLMatrix &C,
                        const TOpenCLMatrix &A,
                        const TOpenCLMatrix &B);
   /** Matrix multiplication of two matrices \p A and \p B^T (transposed) with the
    *  result being written into C.
    */
   static void TransposeMultiply(TOpenCLMatrix &output,
                                 const TOpenCLMatrix &input,
                                 const TOpenCLMatrix &Weights);
   /** In-place Hadamard (element-wise) product of matrices \p A and \p B
    *  with the result being written into \p A.
    */
   static void Hadamard(TOpenCLMatrix &A,
                        const TOpenCLMatrix &B);

   /** Sum columns of (m x n) matrixx \p A and write the results into the first
    * m elements in \p A.
    */
   static void SumColumns(TOpenCLMatrix &B, const TOpenCLMatrix &A);

   /** Compute the sum of all elements in \p A */
   static OpenCLDouble_t Sum(const TOpenCLMatrix &A);
};

} // namespace DNN
} // namespace TMVA

#endif
