from ocelot import Drift, Quadrupole, Undulator, MagneticLattice
from ocelot import rad, ParticleArray

from ocelot.rad import fel_estimator as fel

import copy

__all__ = [
  'SASE',
  'random_beam',
  'random_geometry',

  'N_ELEMENTS',
  'N_CLUSTERS',
  'N_PARTICLES_PER_CLUSTER'
]

def get_lattice(drifts):
  cell = list()
  for ld in drifts[:-1]:
    cell.append(Drift(l=ld))
    cell.append(Quadrupole(l=0.1))

  cell.append(Drift(l=drifts[-1]))
  cell.append(Undulator(lperiod=0.05, nperiods=3, Kx=3, Ky=0.0))

  return cell

N_ELEMENTS = 10

def random_geometry(rng):
  return [
    rng.uniform(2, 10)
    for _ in range(N_ELEMENTS + 1)
  ]


N_CLUSTERS = 5
N_PARTICLES_PER_CLUSTER = 1000

def random_beam(rng):
  pa = ParticleArray(N_PARTICLES_PER_CLUSTER * N_CLUSTERS)
  pa.q_array[:] = 1.25e-10
  pa.E = 0.0325

  for i in range(N_CLUSTERS):
    indx = slice(i * N_PARTICLES_PER_CLUSTER, (i + 1) * N_PARTICLES_PER_CLUSTER)

    cluster_offset_x = rng.normal() * 1e-4
    cluster_offset_y = rng.normal() * 1e-4

    pa.rparticles[0, indx] = rng.normal(size=N_PARTICLES_PER_CLUSTER) * 1e-4 + cluster_offset_x
    pa.rparticles[2, indx] = rng.normal(size=N_PARTICLES_PER_CLUSTER) * 1e-4 + cluster_offset_y

    cluster_offset_px = rng.normal() * 2e-5
    cluster_offset_py = rng.normal() * 2e-5

    pa.rparticles[1, indx] = rng.normal(size=N_PARTICLES_PER_CLUSTER) * 1e-4 + cluster_offset_px
    pa.rparticles[3, indx] = rng.normal(size=N_PARTICLES_PER_CLUSTER) * 1e-4 + cluster_offset_py

    pa.rparticles[4, indx] = rng.normal(size=N_PARTICLES_PER_CLUSTER) * 1e-3

    pa.rparticles[5, indx] = \
      rng.normal(size=N_PARTICLES_PER_CLUSTER) * 1e-4 + 10 * pa.rparticles[4, indx]

  return pa


class SASE(object):
  def __init__(self, particle_array, geometry):
    self.particle_array = particle_array

    self.cell = get_lattice(geometry)

    self.quadrupoles = [
      q
      for q in self.cell
      if q.__class__ == Quadrupole
    ]

  def ndim(self):
    return len(self.quadrupoles)

  def fel_params(self, ks):
    import logging

    logger = logging.getLogger(fel.parray2beam.__module__)
    logger.setLevel(logging.CRITICAL)

    logger = logging.getLogger(fel.__name__)
    logger.setLevel(logging.CRITICAL)

    logger = logging.getLogger(rad.fel.__name__)
    logger.setLevel(logging.CRITICAL)

    lat = MagneticLattice(self.cell)

    p_array = copy.deepcopy(self.particle_array)

    step = 5e-7
    beam = fel.parray2beam(p_array, step=2 * step)

    for q, k in zip(self.quadrupoles, ks):
      q.k1 = k

    lat.update_transfer_maps()
    fel_param = fel.beamlat2fel(beam, lat, smear_m=step)

    return fel_param

  def rho_int(self, ks):
    fel_param = self.fel_params(ks)

    ds = fel_param.s[1] - fel_param.s[0]
    rho_int = sum(fel_param.rho3 * ds)

    return rho_int