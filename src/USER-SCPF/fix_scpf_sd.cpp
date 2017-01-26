/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Frank Uhlig (ICP Stuttgart)
   Sources: Minimization algorithms are basically the ones from min_<name>..
            but adapted to work with LAMMPS groups, no extra degrees of
            freedom
------------------------------------------------------------------------- */

#include <math.h>
#include <mpi.h>
#include "fix_scpf_sd.h"
#include "atom.h"
#include "timer.h"

using namespace LAMMPS_NS;

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8

/* ---------------------------------------------------------------------- */

FixSCPFSD::FixSCPFSD(LAMMPS *lmp, int narg, char **arg) : FixSCPFLS(lmp, narg, arg) {}
FixSCPFSD::~FixSCPFSD() {}

void FixSCPFSD::init()
{
  FixSCPFLS::init();
}

/* ----------------------------------------------------------------------
   minimization via steepest descent
------------------------------------------------------------------------- */

int FixSCPFSD::iterate(int maxiter)
{
  int i,m,n,fail,ntimestep;
  double fdotf;
  double *fatom,*hatom;

  // initialize working vectors

  for (i = 0; i < nvec; i++) h[i] = fvec[i];
  if (nextra_atom)
    for (m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      hatom = hextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++) hatom[i] = fatom[i];
    }
  if (nextra_global)
    for (i = 0; i < nextra_global; i++) hextra[i] = fextra[i];

  niter = neval = 0;

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++niter;

    // line minimization along h from current position x
    // h = downhill gradient direction

    eprevious = ecurrent;
    fail = (this->*linemin)(ecurrent,alpha_final);
    if (fail) return fail;

    // function evaluation criterion

    if (neval >= maxeval) return MAXEVAL;

    // energy tolerance criterion

    if (fabs(ecurrent-eprevious) <
        etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
      return ETOL;

    // force tolerance criterion

    fdotf = fnorm_sqr();
    if (fdotf < ftol*ftol) return FTOL;

    // set new search direction h to f = -Grad(x)

    for (i = 0; i < nvec; i++) h[i] = fvec[i];
    if (nextra_atom)
      for (m = 0; m < nextra_atom; m++) {
        fatom = fextra_atom[m];
        hatom = hextra_atom[m];
        n = extra_nlen[m];
        for (i = 0; i < n; i++) hatom[i] = fatom[i];
      }
    if (nextra_global)
      for (i = 0; i < nextra_global; i++) hextra[i] = fextra[i];

    // output for thermo, dump, restart files

    // if (output->next == ntimestep) {
    //   timer->stamp();
    //   output->write(ntimestep);
    //   timer->stamp(Timer::OUTPUT);
    // }
  }

  return MAXITER;
}
