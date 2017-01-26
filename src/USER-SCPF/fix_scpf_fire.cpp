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
#include <string.h>
#include "fix_scpf.h"
#include "fix_scpf_fire.h"
#include "universe.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "timer.h"
#include "error.h"
#include "comm.h"

using namespace LAMMPS_NS;
using namespace FixConst;

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8

#define DELAYSTEP 5
#define DT_GROW 1.1
#define DT_SHRINK 0.5
#define ALPHA0 0.1
#define ALPHA_SHRINK 0.99
#define TMAX 10.0

/* ---------------------------------------------------------------------- */

FixSCPFFIRE::FixSCPFFIRE(LAMMPS *lmp, int narg, char **arg) : FixSCPF(lmp, narg, arg)
{
  searchflag = 0;
  dtstart = 10*update->dt;
  // FUDO| solves some issues if dt is very long
  // dtstart = update->dt;
  mass_scale = 1.e16; //FU| this is shitty if we have really small masses, should always be changed with fix_modify (or should we rather just use it as a general input parameter???)

  printconv = 0;

}
FixSCPFFIRE::~FixSCPFFIRE() {}

/* ---------------------------------------------------------------------- */

void FixSCPFFIRE::init()
{
  FixSCPF::init();
  dtmax = TMAX * dtstart;
}

/* ---------------------------------------------------------------------- */

void FixSCPFFIRE::setup_style()
{
  int nlocal = atom->nlocal;
  double **v = atom->v;
  int *mask = atom->mask;

  reset_vectors();

  //FUDO| we'll do it for all the velocities here, because we might get spurious stuff in MPI_allreduce otherwise (guess, not knowledge)
  for (int i = 0; i < nlocal; i++)
    if ( mask[i] & groupbit )
      v[i][0] = v[i][1] = v[i][2] = 0.0;

  dt = dtstart;
  alpha = ALPHA0;
  last_negative = 0;
}

/* ----------------------------------------------------------------------
   set current vector lengths and pointers
   called after atoms have migrated
------------------------------------------------------------------------- */

void FixSCPFFIRE::reset_vectors()
{
  // atomic dof

  nvec = 3 * atom->nlocal;
  if (nvec) xvec = atom->x[0];
  if (nvec) fvec = atom->f[0];
}

/* ---------------------------------------------------------------------- */

int FixSCPFFIRE::iterate(int maxiter)
{
  bigint ntimestep;
  double vmax,vdotf,vdotfall,vdotv,vdotvall,fdotf,fdotfall;
  double scale1,scale2;
  double dtvone,dtv,dtf,dtfm;
  int flag,flagall;
  int *mask = atom->mask;

  alpha_final = 0.0;
  niter = 0;
  
  //FUDO| now doing this here, no setup_style anymore
  dt = dtstart;
  alpha = ALPHA0;
  last_negative = 0;

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++niter;

    // vdotfall = v dot f

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

    // if (v dot f) > 0:
    // v = (1-alpha) v + alpha |v| Fhat
    // |v| = length of v, Fhat = unit f
    // if more than DELAYSTEP since v dot f was negative:
    // increase timestep and decrease alpha

    if (vdotfall > 0.0) {
      vdotv = 0.0;
      for (int i = 0; i < nlocal; i++)
        if ( mask[i] & groupbit )
          vdotv += v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
      MPI_Allreduce(&vdotv,&vdotvall,1,MPI_DOUBLE,MPI_SUM,world);

      // sum vdotv over replicas, if necessary
      // this communicator would be invalid for multiprocess replicas

      if (update->multireplica == 1) {
        vdotv = vdotvall;
        MPI_Allreduce(&vdotv,&vdotvall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }

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

      scale1 = 1.0 - alpha;
      if (fdotfall == 0.0) scale2 = 0.0;
      else scale2 = alpha * sqrt(vdotvall/fdotfall);
      for (int i = 0; i < nlocal; i++)
        if ( mask[i] & groupbit ) {
          v[i][0] = scale1*v[i][0] + scale2*f[i][0];
          v[i][1] = scale1*v[i][1] + scale2*f[i][1];
          v[i][2] = scale1*v[i][2] + scale2*f[i][2];
        }

      if (ntimestep - last_negative > DELAYSTEP) {
        dt = MIN(dt*DT_GROW,dtmax);
        alpha *= ALPHA_SHRINK;
      }

    // else (v dot f) <= 0:
    // decrease timestep, reset alpha, set v = 0

    } else {
      last_negative = ntimestep;
      dt *= DT_SHRINK;
      alpha = ALPHA0;
      for (int i = 0; i < nlocal; i++)
        if ( mask[i] & groupbit )
          v[i][0] = v[i][1] = v[i][2] = 0.0;
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
            etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY)) {
            if ( ( printconv ) && ( comm->me == 0) ) printf("converged in %i iterations\n", niter);
          return ETOL;
        }
      } else {
        if (fabs(ecurrent-eprevious) <
            etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) {

            if ( ( printconv ) && ( comm->me == 0) ) printf("converged in %i iterations\n", niter);
            return ETOL;
        }
      }
    }

    // force tolerance criterion
    // sync across replicas if running multi-replica minimization

    if (ftol > 0.0) {
      fdotf = fnorm_sqr();
      if (update->multireplica == 0) {
        if (fdotf < ftol*ftol) {
            if ( ( printconv ) && ( comm->me == 0) ) printf("converged in %i iterations\n", niter);
            return FTOL;
        }
      } else {
        if (fdotf < ftol*ftol) flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) {
            if ( ( printconv ) && ( comm->me == 0 ) ) printf("converged in %i iterations\n", niter);
            return FTOL;
        }
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

int FixSCPFFIRE::modify_param(int narg, char **arg)
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
