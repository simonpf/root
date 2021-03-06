/// \file ROOT/TPadUserCoord.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-07-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TPadUserCoord
#define ROOT7_TPadUserCoord

#include <ROOT/TPadCoord.hxx>

#include <array>
#include <limits>

namespace ROOT {
namespace Experimental {

namespace Detail {

/** \class ROOT::Experimental::Internal::Detail::TPadUserCoordBase
  Base class for user coordinates (e.g. for histograms) used by `TPad` and `TCanvas`.
  */

class TPadUserCoordBase {
private:
   /// Disable copy construction.
   TPadUserCoordBase(const TPadUserCoordBase &) = delete;

   /// Disable assignment.
   TPadUserCoordBase &operator=(const TPadUserCoordBase &) = delete;

protected:
   /// Allow derived classes to default construct a TPadUserCoordBase.
   TPadUserCoordBase() = default;

public:
   virtual ~TPadUserCoordBase();

   /// Convert user coordinates to normal coordinates.
   virtual std::array<TPadCoord::Normal, 2> ToNormal(const std::array<TPadCoord::User, 2> &) const = 0;
};

/** \class ROOT::Experimental::Detail::TPadLinearUserCoord
  The default, linear min/max coordinate system for `TPad`, `TCanvas`.
  */

class TPadLinearUserCoord: public TPadUserCoordBase {
private:
   std::array<double, 2> fMin; ///< (x,y) user coordinate of bottom-left corner
   std::array<double, 2> fMax; ///< (x,y) user coordinate of top-right cornder

   /// For (pos-min)/(max-min) calculations, return a sensible, div-by-0 protected denominator.
   double GetDenominator(int idx) const
   {
      if (fMin[idx] < fMax[idx])
         return std::max(std::numeric_limits<double>::min(), fMin[idx] - fMax[idx]);
      return std::min(-std::numeric_limits<double>::min(), fMin[idx] - fMax[idx]);
   }

public:
   /// Initialize a TPadLinearUserCoord.
   TPadLinearUserCoord(const std::array<double, 2> &min, const std::array<double, 2> &max): fMin(min), fMax(max) {}

   /// Destructor to have a vtable.
   virtual ~TPadLinearUserCoord();

   /// Convert user coordinates to normal coordinates.
   std::array<TPadCoord::Normal, 2> ToNormal(const std::array<TPadCoord::User, 2> &pos) const override
   {
      return {{(pos[0] - fMin[0]) / GetDenominator(0), (pos[1] - fMin[1]) / GetDenominator(1)}};
   }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
