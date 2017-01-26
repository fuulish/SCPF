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
#include <stdlib.h>
#include <string.h>
#include "fix_scpf.h"
#include "atom.h"
#include "atom_vec.h"
#include "domain.h"
#include "comm.h"
#include "fix_minimize.h"
#include "neighbor.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "thermo.h"
#include "timer.h"
#include "memory.h"
#include "error.h"

#include "compute.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSCPF::FixSCPF(LAMMPS *lmp, int narg, char **arg) : FixMinimize(lmp,narg,arg)
{
  dmax = 0.1;
  searchflag = 0;
  linestyle = 1;

  // FUDO| don't think i need those b/c everything is setup in integrator
  // elist_global = elist_atom = NULL;
  // vlist_global = vlist_atom = NULL;

  nextra_global = 0;
  fextra = NULL;

  nextra_atom = 0;
  xextra_atom = fextra_atom = NULL;
  extra_peratom = extra_nlen = NULL;
  extra_max = NULL;
  requestor = NULL;

  external_force_clear = 0;
  
  // FUDO| fix initialization stuff
  if (narg < 6) error->all(FLERR,"Illegal fix SCPF command");

  etol = force->numeric(FLERR,arg[3]);
  ftol = force->numeric(FLERR,arg[4]);
  maxiter = force->inumeric(FLERR,arg[5]);
  maxeval = force->inumeric(FLERR,arg[6]);

  if (etol < 0.0 || ftol < 0.0)
    error->all(FLERR,"Illegal fix SCPF");

  //FU| some defaults that can be changed by fix_modify

  // FUDO| should we take 'em from some global flags? 
  // FUDO| could be problematic with box/relax
  eflag = 1;
  vflag = 0;

  printconv = 0;
  globalfconv = 1;
  first = 1;

  zerovels = 1;

  ndim = domain->dimension;
}

/* ---------------------------------------------------------------------- */

FixSCPF::~FixSCPF()
{
  delete [] fextra;

  memory->sfree(xextra_atom);
  memory->sfree(fextra_atom);
  memory->destroy(extra_peratom);
  memory->destroy(extra_nlen);
  memory->destroy(extra_max);
  memory->sfree(requestor);
}

/* ---------------------------------------------------------------------- */

void FixSCPF::init()
{
  nextra_global = 0;
  delete [] fextra;
  fextra = NULL;

  nextra_atom = 0;
  memory->sfree(xextra_atom);
  memory->sfree(fextra_atom);
  memory->destroy(extra_peratom);
  memory->destroy(extra_nlen);
  memory->destroy(extra_max);
  memory->sfree(requestor);
  xextra_atom = fextra_atom = NULL;
  extra_peratom = extra_nlen = NULL;
  extra_max = NULL;
  requestor = NULL;

  // virial_style:
  // 1 if computed explicitly by pair->compute via sum over pair interactions
  // 2 if computed implicitly by pair->virial_compute via sum over ghost atoms

  if (force->newton_pair) virial_style = 2;
  else virial_style = 1;

  // setup lists of computes for global and per-atom PE and pressure
  // FUDO| will not be modified, but eflag and vflag will be set locally
  // FUDO| need to check on vflag if box/relax is active

  int ifix = modify->find_fix("package_omp");
  if (ifix >= 0) external_force_clear = 1;

  // set flags for arrays to clear in force_clear()

  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;

  // allow pair and Kspace compute() to be turned off via modify flags

  if (force->pair && force->pair->compute_flag) pair_compute_flag = 1;
  else pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag) kspace_compute_flag = 1;
  else kspace_compute_flag = 0;

  // orthogonal vs triclinic simulation box

  triclinic = domain->triclinic;

  // reset reneighboring criteria if necessary

  neigh_every = neighbor->every;
  neigh_delay = neighbor->delay;
  neigh_dist_check = neighbor->dist_check;

  if (neigh_every != 1 || neigh_delay != 0 || neigh_dist_check != 1) {
    if (comm->me == 0)
      error->warning(FLERR,
                     "Resetting reneighboring criteria during minimization");
  }

  neighbor->every = 1;
  neighbor->delay = 0;
  neighbor->dist_check = 1;

  niter = neval = 0;
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void FixSCPF::setup()
{
  // setup extra global dof due to fixes
  // cannot be done in init() b/c update init() is before modify init()

  nextra_global = modify->min_dof();
  if (nextra_global) fextra = new double[nextra_global];

  // compute for potential energy

  // FUDO| there are some issues here with timestep counting, we'll just do it ourselves for now?!
  int id = modify->find_compute("thermo_pe");
  if (id < 0) error->all(FLERR,"Minimization could not find thermo_pe compute");
  pe_compute = modify->compute[id];

  // style-specific setup does two tasks
  // setup extra global dof vectors
  // setup extra per-atom dof vectors due to requests from Pair classes
  // cannot be done in init() b/c update init() is before modify/pair init()

  setup_style();

  // ndoftotal = total dof for entire minimization problem
  // dof for atoms, extra per-atom, extra global

  bigint ndofme = 3 * static_cast<bigint>(atom->nlocal);
  for (int m = 0; m < nextra_atom; m++)
    ndofme += extra_peratom[m]*atom->nlocal;
  MPI_Allreduce(&ndofme,&ndoftotal,1,MPI_LMP_BIGINT,MPI_SUM,world);
  ndoftotal += nextra_global;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists
  // FUDO| needed? already done in integrator setup and doesn't help us much

  // remove these restriction eventually

  if (nextra_global && searchflag == 0)
    error->all(FLERR,
               "Cannot use a damped dynamics min style with fix box/relax");
  if (nextra_atom && searchflag == 0)
    error->all(FLERR,
               "Cannot use a damped dynamics min style with per-atom DOF");

  // atoms may have migrated in comm->exchange()

  reset_vectors();

  // compute all forces

  // FUDO| also not done here, b/c done in integrator setup
  // force->setup();
  // ev_set(update->ntimestep);
  // force_clear();
  // modify->setup_pre_force(vflag);

  if (nextra_atom)
    for (int m = 0; m < nextra_atom; m++)
      requestor[m]->min_xf_get(m);
}

/* ----------------------------------------------------------------------
   perform minimization, calling iterate() for N steps
------------------------------------------------------------------------- */

void FixSCPF::pre_force(int vflag)
{
  reset_vectors();

  ecurrent = energy_force(0);

  fnorm2_init = sqrt(fnorm_sqr());
  fnorminf_init = fnorm_inf();

  if (nextra_global) ecurrent += modify->min_energy(fextra);
  if (output->thermo->normflag) ecurrent /= atom->natoms;

  einitial = ecurrent;

  niter = neval = 0;

  stop_condition = iterate(maxiter);
  stopstr = stopstrings(stop_condition);

  if ( ( printconv ) && ( comm->me == 0 ) ) printf("STOPPING because of %s\n", stopstr);

  if ( ( stop_condition == MAXITER ) && ( comm->me == 0 ) )
      error->warning(FLERR, "Stopping SCPF because of maximum iterations");

  if ( ( stop_condition == MAXEVAL ) && ( comm->me == 0 ) )
      error->warning(FLERR, "Stopping SCPF because of maximum function evaluations");

  force_clear();
}

void FixSCPF::min_post_force(int vflag)
{
    post_force(vflag);
}

void FixSCPF::min_pre_force(int vflag)
{
    pre_force(vflag);
}

//FUDO| these are invoked, because fix is PRE_FORCE-registered
void FixSCPF::setup_pre_force(int vflag)
{
    //FUDO| also calls setup_style()
    setup();

    modify->setup_pre_reverse(eflag, vflag);
    if (kspace_compute_flag)
        force->kspace->setup();
    pre_force(vflag);
}

void FixSCPF::min_setup_pre_force(int vflag)
{
    setup_pre_force(vflag);
}

void FixSCPF::min_setup_post_force(int vflag)
{
    setup_post_force(vflag);
}

void FixSCPF::setup_post_force(int vflag)
{
}

void FixSCPF::post_force(int vflag)
{

    double **v = atom->v;
    int nlocal = atom->nlocal;
    int *mask = atom->mask;

    if ( zerovels )
      for ( int i=0; i<nlocal; i++ )
        if ( mask[i] & groupbit )
          v[i][0] = v[i][1] = v[i][2] = 0.0;

    // comm->forward_comm();
}

int FixSCPF::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  // mask |= INITIAL_INTEGRATE;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSCPF::cleanup()
{
  // FUDO| absolutely not sure in this routine
  modify->post_run();

  // stats for Finish to print

  efinal = ecurrent;
  fnorm2_final = sqrt(fnorm_sqr());
  fnorminf_final = fnorm_inf();

  // reset reneighboring criteria

  neighbor->every = neigh_every;
  neighbor->delay = neigh_delay;
  neighbor->dist_check = neigh_dist_check;

  // delete fix at end of run, so its atom arrays won't persist

  // FUDO| we are that fix
  // modify->delete_fix("MINIMIZE");
  domain->box_too_small_check();
}

/* ----------------------------------------------------------------------
   evaluate potential energy and forces
   may migrate atoms due to reneighboring
   return new energy, which should include nextra_global dof
   return negative gradient stored in atom->f
   return negative gradient for nextra_global dof in fextra
------------------------------------------------------------------------- */

double FixSCPF::energy_force(int resetflag)
{
  // check for reneighboring
  // always communicate since minimizer moved atoms

  // int nflag = 0;
  // comm->forward_comm();

  int nflag = neighbor->decide();

  if (nflag == 0) {
    timer->stamp();
    comm->forward_comm();
    timer->stamp(Timer::COMM);
  } else {
    if (modify->n_pre_exchange) {
      timer->stamp();
      modify->pre_exchange();
      timer->stamp(Timer::MODIFY);
    }
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    if (domain->box_change) {
      domain->reset_box();
      comm->setup();
      if (neighbor->style) neighbor->setup_bins();
    }
    timer->stamp();
    comm->exchange();
    // FUDO| sorting will be done in corresponding integrator routine
    // if (atom->sortfreq > 0 &&
    //     update->ntimestep >= atom->nextsort) atom->sort();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    timer->stamp(Timer::COMM);
    if (modify->n_pre_neighbor) {
      modify->pre_neighbor();
      timer->stamp(Timer::MODIFY);
    }
    neighbor->build();
    timer->stamp(Timer::NEIGH);
  }

  // FUDO| we are using what's set, might need to update vflag settings
  // ev_set(update->ntimestep);
  force_clear();

  timer->stamp();

  // FUDO| cannot do this here
  // if (modify->n_pre_force) {
  //   modify->pre_force(vflag);
  //   timer->stamp(Timer::MODIFY);
  // }

  if (pair_compute_flag) {
    force->pair->compute(eflag,vflag);
    timer->stamp(Timer::PAIR);
  }

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
    timer->stamp(Timer::BOND);
  }

  if (kspace_compute_flag) {
    force->kspace->compute(eflag,vflag);
    timer->stamp(Timer::KSPACE);
  }

  if (modify->n_pre_reverse) {
    modify->pre_reverse(eflag,vflag);
    timer->stamp(Timer::MODIFY);
  }

  if (force->newton) {
    comm->reverse_comm();
    timer->stamp(Timer::COMM);
  }

  // update per-atom minimization variables stored by pair styles

  if (nextra_atom)
    for (int m = 0; m < nextra_atom; m++)
      requestor[m]->min_xf_get(m);

  // fixes that affect minimization

  // FUDO| again, cannot do this here...
  // if (modify->n_post_force) {
  //    timer->stamp();
  //    modify->post_force(vflag);
  //    timer->stamp(Timer::MODIFY);
  // }

  //FUDO| clear forces on particles that are not part of the group

  clear_non_group();

  // compute potential energy of system
  // normalize if thermo PE does

  //FUDO| using our own compute, because below will have issues with ev_tally SOLUTION???
  // double energy = pe_compute->compute_scalar();

  double energy = compute_pe_scalar();
  if (nextra_global) energy += modify->min_energy(fextra);
  if (output->thermo->normflag) energy /= atom->natoms;

  // if reneighbored, atoms migrated
  // if resetflag = 1, update x0 of atoms crossing PBC
  // reset vectors used by lo-level minimizer

  if (nflag) {
    if (resetflag) reset_coords();
    reset_vectors();
  }

  return energy;
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void FixSCPF::clear_non_group()
{
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double **f = atom->f;

    for ( int i=0; i<nlocal; i++ )
      if ( !( mask[i] & groupbit ) )
          f[i][0] = f[i][1] = f[i][2] = 0.;
}

double FixSCPF::compute_pe_scalar()
{
  double scalar = 0.;

  double one = 0.0;
  if (force->pair)
    one += force->pair->eng_vdwl + force->pair->eng_coul;

  if (atom->molecular) {
    if (force->bond) one += force->bond->energy;
    if (force->angle) one += force->angle->energy;
    if (force->dihedral) one += force->dihedral->energy;
    if (force->improper) one += force->improper->energy;
  }

  MPI_Allreduce(&one,&scalar,1,MPI_DOUBLE,MPI_SUM,world);

  if (force->kspace) scalar += force->kspace->energy;

  if (force->pair && force->pair->tail_flag) {
    double volume = domain->xprd * domain->yprd * domain->zprd;
    scalar += force->pair->etail / volume;
  }

  return scalar;
}

void FixSCPF::force_clear()
{
  if (external_force_clear) return;

  // clear global force array
  // if either newton flag is set, also include ghosts

  size_t nbytes = sizeof(double) * atom->nlocal;
  if (force->newton) nbytes += sizeof(double) * atom->nghost;

  if (nbytes) {
    memset(&atom->f[0][0],0,3*nbytes);
    if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
    if (extraflag) atom->avec->force_clear(0,nbytes);
  }
}

/* ----------------------------------------------------------------------
   pair style makes request to add a per-atom variables to minimization
   requestor stores callback to pair class to invoke during min
     to get current variable and forces on it and to update the variable
   return flag that pair can use if it registers multiple variables
------------------------------------------------------------------------- */

int FixSCPF::request(Pair *pair, int peratom, double maxvalue)
{
  int n = nextra_atom + 1;
  xextra_atom = (double **) memory->srealloc(xextra_atom,n*sizeof(double *),
                                             "min:xextra_atom");
  fextra_atom = (double **) memory->srealloc(fextra_atom,n*sizeof(double *),
                                             "min:fextra_atom");
  memory->grow(extra_peratom,n,"min:extra_peratom");
  memory->grow(extra_nlen,n,"min:extra_nlen");
  memory->grow(extra_max,n,"min:extra_max");
  requestor = (Pair **) memory->srealloc(requestor,n*sizeof(Pair *),
                                         "min:requestor");

  requestor[nextra_atom] = pair;
  extra_peratom[nextra_atom] = peratom;
  extra_max[nextra_atom] = maxvalue;
  nextra_atom++;
  return nextra_atom-1;
}

/* ---------------------------------------------------------------------- */

int FixSCPF::modify_param(int narg, char **arg)
{
  if (narg == 0) error->all(FLERR,"Illegal fix_modify command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"dmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      dmax = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"line") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      if (strcmp(arg[iarg+1],"backtrack") == 0) linestyle = 0;
      else if (strcmp(arg[iarg+1],"quadratic") == 0) linestyle = 1;
      else if (strcmp(arg[iarg+1],"forcezero") == 0) linestyle = 2;
      else error->all(FLERR,"Illegal fix_modify command");
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
    } else if (strcmp(arg[iarg],"globalfconv") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix_modify command");
      if (strcmp(arg[iarg+1], "yes") == 0) globalfconv = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) globalfconv = 0;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix_modify command");
  }

  return 2;
}

/* ----------------------------------------------------------------------
   compute and return ||force||_2^2
------------------------------------------------------------------------- */

double FixSCPF::fnorm_sqr()
{
  int i,n;
  double *fatom;

  double local_norm2_sqr = 0.0;
  for (i = 0; i < nvec; i++) local_norm2_sqr += fvec[i]*fvec[i];
  if (nextra_atom) {
    for (int m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++)
        local_norm2_sqr += fatom[i]*fatom[i];
    }
  }

  double norm2_sqr = 0.0;
  MPI_Allreduce(&local_norm2_sqr,&norm2_sqr,1,MPI_DOUBLE,MPI_SUM,world);

  if (nextra_global)
    for (i = 0; i < nextra_global; i++)
      norm2_sqr += fextra[i]*fextra[i];

  return norm2_sqr;
}

/* ----------------------------------------------------------------------
   compute and return ||force||_inf
------------------------------------------------------------------------- */

double FixSCPF::fnorm_inf()
{
  int i,n;
  double *fatom;

  double local_norm_inf = 0.0;
  for (i = 0; i < nvec; i++)
    local_norm_inf = MAX(fabs(fvec[i]),local_norm_inf);
  if (nextra_atom) {
    for (int m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++)
        local_norm_inf = MAX(fabs(fatom[i]),local_norm_inf);
    }
  }

  double norm_inf = 0.0;
  MPI_Allreduce(&local_norm_inf,&norm_inf,1,MPI_DOUBLE,MPI_MAX,world);

  if (nextra_global)
    for (i = 0; i < nextra_global; i++)
      norm_inf = MAX(fabs(fextra[i]),norm_inf);

  return norm_inf;
}

/* ----------------------------------------------------------------------
   possible stop conditions
------------------------------------------------------------------------- */

char *FixSCPF::stopstrings(int n)
{
  const char *strings[] = {"max iterations",
                           "max force evaluations",
                           "energy tolerance",
                           "force tolerance",
                           "search direction is not downhill",
                           "linesearch alpha is zero",
                           "forces are zero",
                           "quadratic factors are zero",
                           "trust region too small",
                           "HFTN minimizer error",
                           "walltime limit reached"};
  return (char *) strings[n];
}
