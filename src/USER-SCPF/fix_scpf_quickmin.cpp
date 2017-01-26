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

#include <mpi.h>
#include <math.h>
#include <string.h>
#include "fix_scpf.h"
#include "fix_scpf_quickmin.h"
#include "universe.h"
#include "atom.h"
#include "force.h"
#include "update.h"
// #include "output.h"
#include "timer.h"
#include "error.h"

using namespace LAMMPS_NS;

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8

#define DELAYSTEP 5

/* ---------------------------------------------------------------------- */

FixSCPFQuickMin::FixSCPFQuickMin(LAMMPS *lmp, int narg, char **arg) : FixSCPF(lmp, narg, arg)
{
  searchflag = 0;
  dtstart = 10*update->dt;
  // FUDO| solves some issues if dt is very long
  // dtstart = update->dt;
  mass_scale = 1.e16; //FU| this is shitty if we have really small masses, should always be changed with fix_modify (or should we rather just use it as a general input parameter???)

  printconv = 0;

}
FixSCPFQuickMin::~FixSCPFQuickMin() {}

/* ---------------------------------------------------------------------- */

void FixSCPFQuickMin::init()
{
  FixSCPF::init();

  dt = update->dt;
  last_negative = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixSCPFQuickMin::setup_style()
{
  double **v = atom->v;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  reset_vectors();

  for (int i = 0; i < nlocal; i++)
    if ( mask[i] & groupbit )
      v[i][0] = v[i][1] = v[i][2] = 0.0;

  dt = dtstart;
  last_negative = 0;
}

/* ----------------------------------------------------------------------
   set current vector lengths and pointers
   called after atoms have migrated
------------------------------------------------------------------------- */

void FixSCPFQuickMin::reset_vectors()
{
  // atomic dof

  nvec = 3 * atom->nlocal;
  if (nvec) xvec = atom->x[0];
  if (nvec) fvec = atom->f[0];
}

/* ----------------------------------------------------------------------
   minimization via QuickMin damped dynamics
------------------------------------------------------------------------- */

int FixSCPFQuickMin::iterate(int maxiter)
{
  bigint ntimestep;
  double vmax,vdotf,vdotfall,fdotf,fdotfall,scale;
  double dtvone,dtv,dtf,dtfm;
  int flag,flagall;
  int *mask = atom->mask;

  niter = 0;

  //FUDO| now doing this here, no setup_style anymore
  dt = dtstart;
  last_negative = 0;

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++niter;

    // zero velocity if anti-parallel to force
    // else project velocity in direction of force

    double **v = atom->v;
    double **f = atom->f;
    int nlocal = atom->nlocal;

    vdotf = 0.0;
    for (int i = 0; i < nlocal; i++)
      if ( mask[i] & groupbit )
        vdotf += v[i][0]*f[i][0] + v[i][1]*f[i][1] + v[i][2]*f[i][2];
    MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,world);

    // sum vdotf over replicas, if necessary
    // this communicator would be invalid for multiprocess replicas

    if (update->multireplica == 1) {
      vdotf = vdotfall;
      MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
    }

    if (vdotfall < 0.0) {
      last_negative = ntimestep;
      for (int i = 0; i < nlocal; i++)
        if ( mask[i] & groupbit )
          v[i][0] = v[i][1] = v[i][2] = 0.0;

    } else {
      fdotf = 0.0;
      for (int i = 0; i < nlocal; i++)
        if ( mask[i] & groupbit )
          fdotf += f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2];
      MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,world);

      // sum fdotf over replicas, if necessary
      // this communicator would be invalid for multiprocess replicas

      if (update->multireplica == 1) {
        fdotf = fdotfall;
        MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }

      if (fdotfall == 0.0) scale = 0.0;
      else scale = vdotfall/fdotfall;
      for (int i = 0; i < nlocal; i++)
        if ( mask[i] & groupbit ) {
          v[i][0] = scale * f[i][0];
          v[i][1] = scale * f[i][1];
          v[i][2] = scale * f[i][2];
      }
    }

    // limit timestep so no particle moves further than dmax

    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;

    dtvone = dt;

    for (int i = 0; i < nlocal; i++)
      if ( mask[i] & groupbit ) {
        vmax = MAX(fabs(v[i][0]),fabs(v[i][1]));
        vmax = MAX(vmax,fabs(v[i][2]));
        if (dtvone*vmax > dmax) dtvone = dmax/vmax;
      }
    MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,world);

    // min dtv over replicas, if necessary
    // this communicator would be invalid for multiprocess replicas

    if (update->multireplica == 1) {
      dtvone = dtv;
      MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,universe->uworld);
    }

    dtf = dtv * force->ftm2v;

    // Euler integration step

    double **x = atom->x;

    if (rmass) {
      for (int i = 0; i < nlocal; i++)
        if ( mask[i] & groupbit ) {
          dtfm = dtf / rmass[i] / mass_scale;
          x[i][0] += dtv * v[i][0];
          x[i][1] += dtv * v[i][1];
          x[i][2] += dtv * v[i][2];
          v[i][0] += dtfm * f[i][0];
          v[i][1] += dtfm * f[i][1];
          v[i][2] += dtfm * f[i][2];
        }
    } else {
      for (int i = 0; i < nlocal; i++)
        if ( mask[i] & groupbit ) {
          dtfm = dtf / mass[type[i]] / mass_scale;
          x[i][0] += dtv * v[i][0];
          x[i][1] += dtv * v[i][1];
          x[i][2] += dtv * v[i][2];
          v[i][0] += dtfm * f[i][0];
          v[i][1] += dtfm * f[i][1];
          v[i][2] += dtfm * f[i][2];
        }
    }

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    // energy tolerance criterion
    // only check after DELAYSTEP elapsed since velocties reset to 0
    // sync across replicas if running multi-replica minimization

    if (etol > 0.0 && ntimestep-last_negative > DELAYSTEP) {
      if (update->multireplica == 0) {
        if (fabs(ecurrent-eprevious) <
            etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          return ETOL;
      } else {
        if (fabs(ecurrent-eprevious) <
            etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return ETOL;
      }
    }

    // force tolerance criterion
    // sync across replicas if running multi-replica minimization

    if (ftol > 0.0) {
      fdotf = fnorm_sqr();
      if (update->multireplica == 0) {
        if (fdotf < ftol*ftol) return FTOL;
      } else {
        if (fdotf < ftol*ftol) flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return FTOL;
      }
    }

    // output for thermo, dump, restart files

    // if (output->next == ntimestep) {
    //   timer->stamp();
    //   output->write(ntimestep);
    //   timer->stamp(Timer::OUTPUT);
    // }
  }

  return MAXITER;
}

int FixSCPFQuickMin::modify_param(int narg, char **arg)
{
  if (narg == 0) error->all(FLERR,"Illegal fix_modify command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"dmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      dmax = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"dt") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      dtstart = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"mass_scale") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      mass_scale = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"dtstart") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      dtstart = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"zerovels") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      if (strcmp(arg[iarg+1], "yes") == 0) zerovels = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) zerovels = 0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"printconv") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      if (strcmp(arg[iarg+1], "yes") == 0) printconv = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) printconv = 0;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix_modify command");
  }

  return 2;
}
