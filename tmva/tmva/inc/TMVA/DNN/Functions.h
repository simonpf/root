// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/06/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Contains function enums for activation and output functions, as //
// well as generic evaluation functions, that delegate the call to //
// the corresponding evaluation kernel.                            //
/////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_FUNCTIONS
#define TMVA_DNN_FUNCTIONS

namespace TMVA
{
namespace DNN
{
//______________________________________________________________________________
//
//  Enum Definitions
//______________________________________________________________________________

/*! Enum that represents layer activation functions. */
enum class EActivationFunction
{
   IDENTITY = 0,
   RELU     = 1,
   SIGMOID  = 2,
   TANH     = 3,
   SYMMRELU = 4,
   SOFTSIGN = 5,
   GAUSS    = 6
};

/*! Enum that represents output functions */
enum class EOutputFunction
{
   IDENTITY = 'I',
   SIGMOID  = 'S'
};

/*! Enum that represents objective functions for the net, i.e. functions
*  that take the output from the last layer in the net together with the
*  truths and return the objective function values that is to be minimized
*  in the training process. */
enum class ELossFunction
{
    CROSSENTROPY     = 'C',
    MEANSQUAREDERROR = 'R'
    };

/*! Enum representing the regularization type applied for a given layer */
enum class ERegularization
{
    NONE = '0',
    L1   = '1',
    L2   = '2'
    };

/* Enum represnting the initialization method used for this layer. */
enum class EInitialization {
    GAUSS    = 'G',
    UNIFORM  = 'U',
    IDENTITY = 'I',
    ZERO = 'Z'
};

//______________________________________________________________________________
//
//  Activation Functions
//______________________________________________________________________________

/*! Apply the given activation function to each value in the given
*  matrix A. */
template<typename Architecture_t>
inline void evaluate(typename Architecture_t::Matrix_t &A,
                    EActivationFunction f)
{
    switch(f)
    {
    case EActivationFunction::IDENTITY : break;
    case EActivationFunction::RELU :     Architecture_t::Relu(A);
        break;
    case EActivationFunction::SIGMOID  :  Architecture_t::Sigmoid(A);
        break;
    case EActivationFunction::TANH     :  Architecture_t::Tanh(A);
        break;
    case EActivationFunction::SYMMRELU :  Architecture_t::SymmetricRelu(A);
        break;
    case EActivationFunction::SOFTSIGN :  Architecture_t::SoftSign(A);
        break;
    case EActivationFunction::GAUSS    :  Architecture_t::Gauss(A);
        break;
    }
}


/*! Compute the first partial derivative of the activation function for
*  the values given in matrix A and write the results into B. */
//______________________________________________________________________________
template<typename Architecture_t>
inline void evaluateDerivative(typename Architecture_t::Matrix_t & B,
                                EActivationFunction f,
                                const typename Architecture_t::Matrix_t & A)
{
    switch(f)
    {
    case EActivationFunction::IDENTITY : Architecture_t::IdentityDerivative(B);
        break;
    case EActivationFunction::RELU     : Architecture_t::ReluDerivative(B, A);
        break;
    case EActivationFunction::SIGMOID  : Architecture_t::SigmoidDerivative(B, A);
        break;
    case EActivationFunction::TANH     : Architecture_t::TanhDerivative(B, A);
        break;
    case EActivationFunction::SYMMRELU : Architecture_t::SymmetricReluDerivative(B, A);
        break;
    case EActivationFunction::SOFTSIGN : Architecture_t::SoftSignDerivative(B, A);
        break;
    case EActivationFunction::GAUSS    : Architecture_t::GaussDerivative(B, A);
        break;
    }
}

//______________________________________________________________________________
//
//  Output Functions
//______________________________________________________________________________

/*! Apply the given output function to each value in the given
*  matrix A. */
template<typename Architecture_t>
inline void evaluate(typename Architecture_t::Matrix_t &A,
                    EOutputFunction f,
                    const typename Architecture_t::Matrix_t &X)
{
    switch(f)
    {
    case EOutputFunction::IDENTITY : break;
    case EOutputFunction::SIGMOID  : Architecture_t::Sigmoid(A, X);
        break;
    }
}

//______________________________________________________________________________
//
//  Loss Functions
//______________________________________________________________________________

/*! Compute the value of the objective function f for given activations
*  of the ouput layer and the truth Y. */
template<typename Architecture_t>
inline auto evaluate(ELossFunction f,
                    const typename Architecture_t::Matrix_t & Y,
                    const typename Architecture_t::Matrix_t & output)
-> decltype(Architecture_t::CrossEntropy(Y,output))
{
    switch(f)
    {
    case ELossFunction::CROSSENTROPY :
        return Architecture_t::CrossEntropy(Y, output);
    case ELossFunction::MEANSQUAREDERROR :
        return Architecture_t::MeanSquaredError(Y, output);
    }
    return 0.0;
}

/*! Compute the gradient of the given output function f for given activations
*  output of the output layer and truth Y and write the results into dY. */
//______________________________________________________________________________
template<typename Architecture_t>
inline void evaluateGradients(typename Architecture_t::Matrix_t & dY,
                                ELossFunction f,
                                const typename Architecture_t::Matrix_t &Y,
                                const typename Architecture_t::Matrix_t &output)
{
    switch(f)
    {
    case ELossFunction::CROSSENTROPY :
        Architecture_t::CrossEntropyGradients(dY, Y, output);
        break;
    case ELossFunction::MEANSQUAREDERROR :
        Architecture_t::MeanSquaredErrorGradients(dY, Y, output);
        break;
    }
}


//______________________________________________________________________________
//
// Regularization
//______________________________________________________________________________

/*! Evaluate the regularization functional for a given weight matrix. */
template<typename Architecture_t>
inline auto regularization(const typename Architecture_t::Matrix_t &A,
                    ERegularization R)
-> decltype(Architecture_t::L1Regularization(A))
{
    switch(R)
    {
    case ERegularization::NONE :
        return 0.0;
    case ERegularization::L1 :
        return Architecture_t::L1Regularization(A);
    case ERegularization::L2 :
        return Architecture_t::L2Regularization(A);
    }
    return 0.0;
}

/*! Add the regularization gradient corresponding to weight matrix W, to
*  the matrix A. */
//______________________________________________________________________________
template<typename Architecture_t>
inline void addRegularizationGradients(typename Architecture_t::Matrix_t &A,
                                       const typename Architecture_t::Matrix_t &W,
                                       typename Architecture_t::Scalar_t weightDecay,
                                       ERegularization R)
{
    switch(R)
    {
    case ERegularization::NONE :
        break;
    case ERegularization::L1 :
        Architecture_t::AddL1RegularizationGradients(A, W, weightDecay);
        break;
    case ERegularization::L2 :
        Architecture_t::AddL2RegularizationGradients(A, W, weightDecay);
        break;
    }
}

//______________________________________________________________________________
//
// Initialization
//______________________________________________________________________________

template<typename Architecture_t>
inline void initialize(typename Architecture_t::Matrix_t & A,
                       EInitialization m)
{
   switch(m) {
   case EInitialization::GAUSS    : Architecture_t::InitializeGauss(A);
       break;
   case EInitialization::UNIFORM  : Architecture_t::InitializeUniform(A);
       break;
   case EInitialization::IDENTITY : Architecture_t::InitializeIdentity(A);
       break;
   case EInitialization::ZERO     : Architecture_t::InitializeZero(A);
       break;
   }
}

//______________________________________________________________________________
//
// Initialization
//______________________________________________________________________________

template<typename Architecture_t>
inline void Flops(EActivationFunction f)
{
   size_t flops;
   switch(f) {
   case EActivationFunction::Identity      : flops =  0;
       break;
   case EActivationFunction::Relu          : flops =  2;
       break;
   case EActivationFunction::Sigmoid       : flops = 36;
       break;
   case EActivationFunction::Tanh          : flops = 42;
       break;
   case EActivationFunction::SymmetricRelu : flops =  2;
       break;
   case EActivationFunction::SoftSign      : flops = 16;
       break;
   case EActivationFunction::Gauss         : flops = 26;
       break;
   }
}

} // namespace DNN
} // namespace TMVA

#endif
