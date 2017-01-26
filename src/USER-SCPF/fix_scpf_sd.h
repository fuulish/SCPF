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

FixStyle(scpf/sd,FixSCPFSD)    // This registers this fix class with LAMMPS.

#else

#ifndef LMP_FIX_SCPF_SD_H
#define LMP_FIX_SCPF_SD_H

#include "fix.h"
#include "fix_scpf_ls.h"

namespace LAMMPS_NS {

class FixSCPFSD : public FixSCPFLS {
 public:
  FixSCPFSD(class LAMMPS *, int, char **);
  ~FixSCPFSD();
  void init();
  // void setup_style();
  int iterate(int);
};

}

#endif
#endif
