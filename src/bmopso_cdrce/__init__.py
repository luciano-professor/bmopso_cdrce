"""Package bmopso_cdrce: Binary Multi-Objective Particle Swarm Optimization with Crowding Distance, Roulette Wheel, and Catfish Effect for pymoo.

Implements the BMOPSO-CDRCE algorithm, combining:
1. BMOPSO-CDR (Souza et al., ICTAI 2011; Santana et al., 2009; Coello Coello et al., 2004; Kennedy & Eberhart, 1997).
2. Catfish Effect Operator (Chuang et al., 2011; Souza et al., WTF 2014, DOI: 10.5753/wtf.2014.22943).

References
----------
- Souza, L. S., Miranda, P. B. C., Prudêncio, R. B. C., & Barros, F. A. (2011).
  "A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on
  Functional Requirements Coverage and Execution Effort", ICTAI 2011. DOI: 10.1109/ICTAI.2011.45
- Souza, L. S., Prudêncio, R. B. C., & Barros, F. A. (2014).
  "Multi-Objective Test Case Selection: A study of the influence of the Catfish effect on PSO based strategies",
  Anais do XV Workshop de Testes e Tolerância a Falhas (WTF 2014), SBC. DOI: 10.5753/wtf.2014.22943
- Santana, R. A., Pontes, M. R., & Bastos-Filho, C. J. A. (2009).
  "A Multiple Objective Particle Swarm Optimization Approach Using Crowding Distance and Roulette Wheel",
  ICNC 2009. DOI: 10.1109/ICNC.2009.610
- Chuang, L. Y., Tsai, S. W., & Yang, C. H. (2011).
  "Improved binary particle swarm optimization using catfish effect for feature selection",
  Expert Systems with Applications, 38(10), 12699-12707. DOI: 10.1016/j.eswa.2011.04.057
- Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004).
  "Handling multiple objectives with particle swarm optimization",
  IEEE Transactions on Evolutionary Computation, 8(3), 256-279.
- Deb, K. (2002).
  "Multi-Objective Optimization using Evolutionary Algorithms", John Wiley & Sons.
- Kennedy, J., & Eberhart, R. C. (1997).
  "A discrete binary version of the particle swarm algorithm", IEEE SMC 1997.
"""

from .algorithms.bmopso_cdrce import BMOPSO_CDRCE
from .operators.catfish import (
    apply_catfish_effect,
    generate_extreme_binary_positions,
    select_random_particles,
)
from .operators.mutation import apply_mutation
from .operators.pbest import update_personal_bests
from .operators.sampling import sample_binary_positions, sigmoid
from .operators.velocity import update_velocity
from .util.archive import NonDominatedArchive
from .util.diversity import calc_crowding_distance, calc_crowding_roulette_probabilities
from .util.dominance import dominates, find_non_dominated_constrained

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "BMOPSO_CDRCE",
    "NonDominatedArchive",
    "apply_catfish_effect",
    "apply_mutation",
    "calc_crowding_distance",
    "calc_crowding_roulette_probabilities",
    "dominates",
    "find_non_dominated_constrained",
    "generate_extreme_binary_positions",
    "sample_binary_positions",
    "select_random_particles",
    "sigmoid",
    "update_personal_bests",
    "update_velocity",
]
