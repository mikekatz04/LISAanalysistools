from __future__ import annotations

import warnings
from typing import Dict, Any

from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer
from eryn.utils.periodic import PeriodicContainer

from .base import Prior
from .joint import JointPrior
from .gfpriors import LISAPriorDict


class BaseSourceConfig:
    """Unified infrastructure for LISA source priors and transformations.

    Parses a LISAPriorDict to automatically construct Eryn's ProbDistContainer,
    TransformContainer, and PeriodicContainer.

    Args:
        prior_dict (LISAPriorDict): The dictionary of initialized Prior objects.
        source_name (str): Identifier for the source (e.g., 'mbhb', 'gb').
        use_cupy (bool): Whether to use CuPy for GPU acceleration.
        return_gpu (bool): Whether the priors should return arrays on the GPU.
    """

    def __init__(
        self,
        prior_dict: LISAPriorDict,
        source_name: str,
        use_cupy: bool = False,
        return_gpu: bool = False,
    ):
        self.source_name = source_name
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu

        self.sampled_priors = {}
        self.fixed_priors = {}

        # Categorize priors and propagate device usage
        for key, prior in prior_dict.items():
            prior.use_cupy = self.use_cupy
            prior.return_gpu = self.return_gpu

            # Identify fixed parameters
            if prior.__class__.__name__ == "DeltaFunction":
                self.fixed_priors[key] = prior
            else:
                self.sampled_priors[key] = prior

        self.priors = {self.source_name: self._build_prob_dist_container()}
        self.transform = self._build_transform_container()
        self.periodic = self._build_periodic_container()

    def _build_prob_dist_container(self) -> ProbDistContainer:
        """Constructs the primary probability distribution container."""
        priors_in = {}
        for prior in self.sampled_priors.values():
            if isinstance(prior, JointPrior):
                priors_in[prior.names] = prior
            else: # Prior class
                priors_in[prior.name] = prior

        return ProbDistContainer(
            priors_in, 
            use_cupy=self.use_cupy, 
            return_gpu=self.return_gpu
        )

    def _build_transform_container(self) -> "TransformContainer":
        """Dynamically builds the transformation and fill-value container."""
        input_basis = []
        output_basis = []
        parameter_transforms = {}
        key_map = {}
        fill_dict = {}

        for prior in self.sampled_priors.values():
            if isinstance(prior, JointPrior):
                input_basis.extend(prior.names)
                output_basis.extend(prior.names_phys)
                
                # Map keys and register N-dimensional multi-transforms
                for n, np_name in zip(prior.names, prior.names_phys):
                    if n != np_name:
                        key_map[n] = np_name
                
                if getattr(prior, "inverse_transform_nd", None) is not None:
                    parameter_transforms[prior.names_phys] = prior.inverse_transform_nd

            else:
                input_basis.append(prior.name)
                output_basis.append(prior.name_phys)

                if prior.name != prior.name_phys:
                    key_map[prior.name] = prior.name_phys

                if prior.inverse_transform is not None:
                    parameter_transforms[prior.name_phys] = prior.inverse_transform

        for prior in self.fixed_priors.values():
            output_basis.append(prior.name_phys)
            fill_dict[prior.name_phys] = prior.peak  # Inject constant value

        return TransformContainer(
            input_basis=input_basis,
            output_basis=output_basis,
            parameter_transforms=parameter_transforms if parameter_transforms else None,
            fill_dict=fill_dict if fill_dict else None,
            key_map=key_map if key_map else None,
        )

    def _build_periodic_container(self) -> "PeriodicContainer":
        """Dynamically builds the periodic boundary container."""
        periodic_in = {}

        for prior in self.sampled_priors.values():
            if isinstance(prior, JointPrior):
                for i, boundary in enumerate(prior.boundaries):
                    if boundary == "periodic":
                        warnings.warn(
                            f"Periodic boundaries for JointPriors ({prior.names[i]}) "
                            "are not yet fully supported via automated widths."
                        )
            else:
                if getattr(prior, "boundary", None) == "periodic":
                    # The wrap period is the width of the sampling prior
                    periodic_in[prior.name] = prior.maximum - prior.minimum

        if not periodic_in:
            return PeriodicContainer(periodic={})

        return PeriodicContainer(periodic={self.source_name: periodic_in})
        
        
class GBConfig(BaseSourceConfig):
    """
    Configuration for Galactic Binary sources in LISA data analysis.
    
    If no prior file is provided, falls back to the standard Mojito-light 
    LISA data challenge prior for non-eccentric galactic binaries.
    """

    def __init__(
        self, 
        prior_file: str | None = None, 
        use_cupy: bool = False, 
        return_gpu: bool = False
    ):
        if prior_file is not None:
            # Safely parse the user-provided .prior text file
            prior_dict = LISAPriorDict.from_file(prior_file)
        else:
            # Use the built-in default configuration
            prior_dict = self._get_default_prior_dict()
        
        super().__init__(
            prior_dict=prior_dict,
            source_name="gb",
            use_cupy=use_cupy,
            return_gpu=return_gpu
        )

    @classmethod
    def _get_default_prior_dict(cls) -> "LISAPriorDict":
        """
        Returns the standard Mojito-light LISA data challenge priors for GBs.
        """
        gb_prior_text = """
            # The prior for a non-eccentric galactic binary used for the Mojito-light LISA data challenge
            # Lines with # are ignored.
            logA = LogUniform(minimum=10**(-23.2), maximum=1e-20, name="logA", name_phys="A", latex_label=r"\log \mathcal{A}")
            f0 = Uniform(minimum=1e-4, maximum=2.1e-2, name="f0", unit="Hz", latex_label=r"f_0")
            fdot = Uniform(minimum=-1e-13, maximum=1e-13, name="fdot", latex_label=r"\dot{f}")
            fddot = DeltaFunction(peak=0.0, name="fddot", name_phys="fddot", latex_label=r"\ddot{f}")
            phi0 = Uniform(minimum=0.0, maximum=2*np.pi, name="phi0", boundary="periodic", latex_label=r"\phi_0")
            cos_inc = UniformInCosine(name="cos_inc", name_phys="inc", latex_label=r"\cos \iota")
            psi = Uniform(minimum=0.0, maximum=np.pi, name="psi", boundary="periodic", latex_label=r"\psi")
            ra = Uniform(minimum=0.0, maximum=2*np.pi, name="ra", boundary="periodic", latex_label=r"\alpha")
            sin_dec = UniformInSine(name="sin_dec", name_phys="dec", latex_label=r"\sin \delta")
        """
        # (Note: Assumes LogUniform is mapped to Log10Uniform, and CosineUniform to UniformInCosine)
        return LISAPriorDict.from_string(gb_prior_text)

    @classmethod
    def default(
        cls, 
        use_cupy: bool = False, 
        return_gpu: bool = False
    ) -> "GBConfig":
        """
        Explicitly instantiate the default Galactic Binary configuration.
        """
        return cls(prior_file=None, use_cupy=use_cupy, return_gpu=return_gpu)