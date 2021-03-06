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

FixStyle(scpf/cg,FixSCPFCG)    // This registers this fix class with LAMMPS.

#else

#ifndef LMP_FIX_SCPF_CG_H
#define LMP_FIX_SCPF_CG_H

#include "fix.h"
#include "fix_scpf_ls.h"

namespace LAMMPS_NS {

class FixSCPFCG : public FixSCPFLS {
 public:
  FixSCPFCG(class LAMMPS *, int, char **);
  ~FixSCPFCG();
  void init();
  // void setup_style();
  int iterate(int);
};

}

#endif
#endif
