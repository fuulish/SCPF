/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(scpf/quickmin,FixSCPFQuickMin)    // This registers this fix class with LAMMPS.

#else

#ifndef LMP_FIX_SCPF_QUICKMIN_H
#define LMP_FIX_SCPF_QUICKMIN_H

#include "fix.h"
#include "fix_scpf.h"

namespace LAMMPS_NS {

class FixSCPFQuickMin : public FixSCPF {
 public:
  FixSCPFQuickMin(class LAMMPS *, int, char **);
  ~FixSCPFQuickMin();
  void init();
  void setup_style();
  void reset_vectors();
  int iterate(int);
  int modify_param(int narg, char **arg);

 private:
  double dt,dtstart,dtmax;
  double alpha;
  bigint last_negative;
  double mass_scale;
};

}

#endif
#endif
