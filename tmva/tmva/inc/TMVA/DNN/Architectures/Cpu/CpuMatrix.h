// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////
// Definition of the CpuMatrix class used to represent  //
// weight and bias matrices in neural nets.             //
//////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CPU_CPUMATRIX
#define TMVA_DNN_ARCHITECTURES_CPU_CPUMATRIX

#include <cstddef>
#include <vector>
#include "tbb/tbb.h"

#include "TMatrix.h"
#include "CpuBuffer.h"

namespace TMVA
{
namespace DNN
{

/** The TCpuMatrix class.
 *
 * Matrix class for multi-threaded CPU architectures. Uses the TCpuBuffer
 * class to store the matrices in column-major format for compatibility with
 * BLAS. Provides Map and MapFrom member functions to simplify the application of
 * activation functions and derivatives to matrices.
 *
 * Copying and assignment of TCpuMatrix objects only performs shallow copies, i.e.
 * copying is fast and the resulting objects share the element data.
 *
 * \tparam AFloat The floating point type used to represent the matrix elements.
 */
//______________________________________________________________________________
template<typename AFloat>
class TCpuMatrix
{
private:

   static std::vector<AFloat> fOnes;  ///< Vector filled with ones used for BLAS calls.

   TCpuBuffer<AFloat> fBuffer; ///< The buffer holding the matrix elements
                              ///< in column-major format.
   size_t            fNCols;
   size_t            fNRows;

public:

   /** Returns pointer to a vector holding only ones with a guaranteed length
    *  of the number of columns of every instantiated CpuMatrix object. */
   static const AFloat * GetOnePointer() {return fOnes.data();}

   /** Construct matrix and allocate space for its elements. */
   TCpuMatrix(size_t nRows, size_t nCols);
   /** Construct a TCpuMatrix object by (deeply) copying from a
    *  TMatrixT<Double_t> matrix. */
   TCpuMatrix(const TMatrixT<Double_t> &);
   /** Construct a m-times-n matrix from the given buffer. The size must of
    *  course match. */
   TCpuMatrix(const TCpuBuffer<AFloat> &buffer, size_t m, size_t n);

   TCpuMatrix(const TCpuMatrix  &)             = default;
   TCpuMatrix(      TCpuMatrix &&)             = default;
   TCpuMatrix & operator=(const TCpuMatrix &)  = default;
   TCpuMatrix & operator=(TCpuMatrix &&)       = default;
   ~TCpuMatrix()                               = default;

   /** Convert to a TMatrixT<Double_t> object. Performs a deep copy of the matrix
    *  elements. */
   operator TMatrixT<Double_t>() const;

   /** Map the given function over the matrix elements. Executed in parallel
    *  using tbb. */
   template <typename Function_t>
   void Map(Function_t &f);

   /** Same as maps but takes the input values from the matrix \p A and writes
    *  the results in this matrix. */
   template <typename Function_t>
   void MapFrom(Function_t &f, const TCpuMatrix & A);

   size_t GetNrows() const {return fNRows;}
   size_t GetNcols() const {return fNCols;}
   size_t GetNElements() const {return fNRows * fNCols;}

   /** Return matrix element in row \p i and column \p j. */
   AFloat   operator()(size_t i, size_t j) const {return fBuffer[j * fNRows + i];}
   AFloat & operator()(size_t i, size_t j)       {return fBuffer[j * fNRows + i];}

   /** Return raw pointer to the elements stored contiguously in column-major
    *  order. */
   AFloat *       GetRawDataPointer()        {return fBuffer;}
   const AFloat * GetRawDataPointer()  const {return fBuffer;}

private:

   void Initialize();

};

// Inline Functions.
//______________________________________________________________________________
template<typename AFloat>
template<typename Function_t>
inline void TCpuMatrix<AFloat>::Map(Function_t &f)
{
   AFloat __restrict__ *data = GetRawDataPointer();

   auto fRange = [data, &f](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         data[i] = f(data[i]);
      }
   };

   tbb::blocked_range<size_t> range(0, GetNElements());
   parallel_for(range, fRange);
}

template<typename AFloat>
template<typename Function_t>
inline void TCpuMatrix<AFloat>::MapFrom(Function_t &f, const TCpuMatrix &A)
{
         AFloat __restrict__ *dataB = GetRawDataPointer();
   const AFloat __restrict__ *dataA = A.GetRawDataPointer();

   auto fRange = [&dataB, &dataA, &f](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
         size_t rangeEnd   = range.end();

         for (size_t i = rangeBegin; i != rangeEnd; ++i) {
            dataB[i] = f(dataA[i]);
         }
   };

   tbb::blocked_range<size_t> range(0, GetNElements());
   parallel_for(range, fRange);
}

} // namespace DNN
} // namespace TMVA

#endif
