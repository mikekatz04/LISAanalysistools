from __future__ import annotations

import warnings
from typing import Dict

from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer
from eryn.utils.periodic import PeriodicContainer

from .base import Prior
from .joint import JointPrior
from .gfpriors import LISAPriorDict
from ...sources.utils import ecliptic_to_icrs, mT_Q, gpc_to_mpc

from bbhx.utils.transform import mT_q, LISA_to_SSB


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
        self.prior_dict = prior_dict
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

    def _build_transform_container(self) -> TransformContainer | None:
        """Dynamically builds the transformation and fill-value container."""
        input_basis = []
        output_basis = []
        parameter_transforms = {}
        key_map = {}
        fill_dict = {}

        def enforce_device(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if not self.return_gpu and hasattr(result, "get"):
                    return result.get() # Force it back to CPU
                return result
            return wrapper
        
        for prior in self.prior_dict.values():

            is_fixed = type(prior).__name__ == "DeltaFunction"
            
            if isinstance(prior, JointPrior):
                output_basis.extend(prior.names_phys)
                
                if not is_fixed:
                    input_basis.extend(prior.names)
                    for n, np_name in zip(prior.names, prior.names_phys):
                        if n != np_name:
                            key_map[n] = np_name
                    
                    if getattr(prior, "inverse_transform_nd", None) is not None:
                        parameter_transforms[prior.names_phys] = enforce_device(prior.inverse_transform_nd)
                else:
                    for np_name, peak in zip(prior.names_phys, getattr(prior, 'peak')):
                        fill_dict[np_name] = peak

            else:
                output_basis.append(prior.name_phys)

                if not is_fixed:
                    input_basis.append(prior.name)
                    
                    if prior.name != prior.name_phys:
                        key_map[prior.name] = prior.name_phys

                    if prior.inverse_transform is not None:
                        parameter_transforms[prior.name_phys] = enforce_device(prior.inverse_transform)
                else:
                    # Inject the fixed value
                    fill_dict[prior.name_phys] = prior.peak

        if not parameter_transforms and not key_map and not fill_dict:
            return None

        return TransformContainer(
            input_basis=input_basis,
            output_basis=output_basis,
            parameter_transforms=parameter_transforms if parameter_transforms else None,
            fill_dict=fill_dict if fill_dict else None,
            key_map=key_map if key_map else None,
        )

    def _build_periodic_container(self) -> PeriodicContainer:
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
        
        assert self.transform is not None and self.transform.input_basis is not None, "TransformContainer must be built before PeriodicContainer."
        return PeriodicContainer(
            periodic={self.source_name: periodic_in},
            key_order={self.source_name: tuple(self.transform.input_basis)}
        )
    
    def update_prior_kwargs(self, param_key: str | tuple, **kwargs) -> None:
        """
        Update attributes of a specific prior object and automatically 
        rebuild the Eryn containers so fixed values and periods stay synced.
        """
        # Find the target prior in either sampled or fixed dictionaries
        prior = self.sampled_priors.get(param_key) or self.fixed_priors.get(param_key)
        
        if prior is None:
            raise KeyError(f"[{self.source_name}] Prior key {param_key} not found.")
        
        # Safely update the attributes (e.g., minimum, maximum, use_cupy, return_gpu)
        for k, v in kwargs.items():
            setattr(prior, k, v)
            
        # Re-build Eryn containers
        self.priors = {self.source_name: self._build_prob_dist_container()}
        self.transform = self._build_transform_container()
        self.periodic = self._build_periodic_container()
        
        
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
        prior_dict = (
            LISAPriorDict.from_file(prior_file) 
            if prior_file is not None 
            else self._get_default_prior_dict()
        )
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
            logA = LogUniform(minimum=10**(-23.2), maximum=1e-20, name="logA", name_phys="A", latex_label=r"\log \mathcal{A}")
            f0 = Uniform(minimum=1e-4, maximum=2.1e-2, name="f0", unit="Hz", latex_label=r"f_0")
            fdot = Uniform(minimum=-1e-13, maximum=1e-13, name="fdot", latex_label=r"\dot{f}")
            fddot = DeltaFunction(peak=0.0, name="fddot", name_phys="fddot", latex_label=r"\ddot{f}")
            phi0 = Uniform(minimum=0.0, maximum=2*np.pi, name="phi0", boundary="periodic", latex_label=r"\phi_0")
            cos_inc = CosineUniform(name="cos_inc", name_phys="inc", latex_label=r"\cos \iota")
            psi = Uniform(minimum=0.0, maximum=np.pi, name="psi", boundary="periodic", latex_label=r"\psi")
            ra = Uniform(minimum=0.0, maximum=2*np.pi, name="ra", boundary="periodic", latex_label=r"\alpha")
            sin_dec = SineUniform(name="sin_dec", name_phys="dec", latex_label=r"\sin \delta")
        """
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
    
    
    
class PSDAnalyticalConfig(BaseSourceConfig):
    """Configuration for LISA PSD parameters."""

    def __init__(
        self, 
        prior_file: str | None = None, 
        use_cupy: bool = False, 
        return_gpu: bool = False
    ):
        prior_dict = (
            LISAPriorDict.from_file(prior_file) 
            if prior_file is not None 
            else self._get_default_prior_dict()
        )
        super().__init__(
            prior_dict=prior_dict,
            source_name="psd",
            use_cupy=use_cupy,
            return_gpu=return_gpu
        )

    @classmethod
    def _get_default_prior_dict(cls) -> "LISAPriorDict":
        psd_prior_text = """
            log10_S_oms = Log10Uniform(minimum=1e-12, maximum=1e-10, name="log10_S_oms", name_phys="S_oms", latex_label=r"\log_{10} S_{\rm oms}")
            log10_S_tm = Log10Uniform(minimum=1e-16, maximum=1e-13, name="log10_S_tm", name_phys="S_tm", latex_label=r"\log_{10} S_{\rm tm}")
        """
        return LISAPriorDict.from_string(psd_prior_text)

    @classmethod
    def default(cls, use_cupy: bool = False, return_gpu: bool = False) -> "PSDAnalyticalConfig":
        return cls(None, use_cupy, return_gpu)



class GalForConfig(BaseSourceConfig):
    """Configuration for Galactic Foreground spectral parameters."""

    def __init__(
        self, 
        prior_file: str | None = None, 
        use_cupy: bool = False, 
        return_gpu: bool = False
    ):
        prior_dict = (
            LISAPriorDict.from_file(prior_file) 
            if prior_file is not None 
            else self._get_default_prior_dict()
        )
        super().__init__(
            prior_dict=prior_dict,
            source_name="galfor",
            use_cupy=use_cupy,
            return_gpu=return_gpu
        )

    @classmethod
    def _get_default_prior_dict(cls) -> "LISAPriorDict":
        galfor_prior_text = """
            log10_Amp = Log10Uniform(minimum=1e-46, maximum=1e-43, name="log10_Amp", name_phys="Amp", latex_label=r"\log_{10} A_{\rm gal}")
            alpha = Uniform(minimum=1.0, maximum=8.0, name="alpha", name_phys="alpha", latex_label=r"\alpha_{\rm gal}")
            log10_f1 = Log10Uniform(minimum=1e-3, maximum=1e-2, name="log10_f1", name_phys="f1", latex_label=r"\log_{10} f_1")
            log10_fknee = Log10Uniform(minimum=1e-3, maximum=1e-1, name="log10_fknee", name_phys="fknee", latex_label=r"\log_{10} f_{\rm knee}")
            log10_f2 = Log10Uniform(minimum=1e-4, maximum=1e-2, name="log10_f2", name_phys="f2", latex_label=r"\log_{10} f_2")
        """
        return LISAPriorDict.from_string(galfor_prior_text)

    @classmethod
    def default(cls, use_cupy: bool = False, return_gpu: bool = False) -> "GalForConfig":
        return cls(None, use_cupy, return_gpu)
    
    

class MBHConfig(BaseSourceConfig):
    def __init__(self, prior_file: str | None = None, use_cupy: bool = False, return_gpu: bool = False):
        prior_dict = (
            LISAPriorDict.from_file(prior_file) 
            if prior_file is not None 
            else self._get_default_prior_dict()
        )
        super().__init__(
            prior_dict=prior_dict,
            source_name="mbh",
            use_cupy=use_cupy,
            return_gpu=return_gpu
        )

    @staticmethod
    def _get_default_prior_dict() -> LISAPriorDict:
        mbh_prior_text = """
            logM = LogUniform(minimum=11.51292, maximum=18.42068, name="logM", inverse_transform=np.exp)
            Q = LogUniform(minimum=1., maximum=10., name="Q")
            s1z = Uniform(minimum=-0.99999999, maximum=0.99999999, name="s1z")
            s2z = Uniform(minimum=-0.99999999, maximum=0.99999999, name="s2z")
            dist = Uniform(minimum=1.0, maximum=150.0, name="dist", inverse_transform=gpc_to_mpc, latex_label=r"d_L \, [\mathrm{Mpc}]")
            phi_ref = Uniform(minimum=0.0, maximum=2 * np.pi, name="phi_ref", boundary="periodic")
            cos_iota = CosineUniform(name="cos_iota", name_phys="iota", latex_label=r"\cos \iota")
            psi = Uniform(minimum=0.0, maximum=np.pi, name="psi", boundary="periodic", latex_label=r"\psi")
            lam = Uniform(minimum=0.0, maximum=2 * np.pi, name="lam", boundary="periodic", latex_label=r"\lambda")
            sin_beta = SineUniform(name="sin_beta", name_phys="beta", latex_label=r"\sin \beta")
            t_plunge = Uniform(minimum=0.0, maximum=1.0, name="t_plunge")
        """
        return LISAPriorDict.from_string(mbh_prior_text)

    # def get_transform_container(self) -> TransformContainer:
    #     tc = super().get_transform_container()
    #     if tc.base_transforms is None: tc.base_transforms = {"single_param": {}, "mult_param": {}}
    #     if "mult_param" not in tc.base_transforms: tc.base_transforms["mult_param"] = {}
            
    #     tc.base_transforms["mult_param"].update({
    #         ("logM", "Q"): mT_Q,
    #         ("t_plunge", "lam", "sin_beta", "psi"): LISA_to_SSB,
    #         ("lam", "sin_beta", "psi"): ecliptic_to_icrs,
    #     })
    #     return tc
    
    @classmethod
    def default(cls, use_cupy: bool = False, return_gpu: bool = False) -> "MBHConfig":
        return cls(None, use_cupy, return_gpu)
    
    
    
class EMRIConfig(BaseSourceConfig):
    def __init__(self, prior_file: str | None = None, use_cupy: bool = False, return_gpu: bool = False):
        if prior_file is not None:
            prior_dict = LISAPriorDict.from_file(prior_file)
        else:
            prior_dict = self._get_default_prior_dict()

        super().__init__(
            prior_dict=prior_dict,
            source_name="emri",
            use_cupy=use_cupy,
            return_gpu=return_gpu
        )

    @staticmethod
    def _get_default_prior_dict() -> LISAPriorDict:
        emri_prior_text = """
            logm1 = LogUniform(minimum=13.12236, maximum=15.42494, name="logm1", name_phys="m1")
            m2 = Uniform(minimum=1.0, maximum=100.0, name="m2")
            a = Uniform(minimum=0.01, maximum=0.999, name="a")
            p0 = Uniform(minimum=5.0, maximum=100.0, name="p0")
            e0 = Uniform(minimum=0.001, maximum=0.8, name="e0")
            dist = Uniform(minimum=0.01, maximum=100.0, name="dist")
            cos_qS = CosineUniform(minimum=-0.99999, maximum=0.99999, name="cos_qS", name_phys="qS")
            phiS = Uniform(minimum=0.0, maximum=2 * np.pi, name="phiS", boundary="periodic")
            cos_qK = CosineUniform(minimum=-0.99999, maximum=0.99999, name="cos_qK", name_phys="qK")
            phiK = Uniform(minimum=0.0, maximum=2 * np.pi, name="phiK", boundary="periodic")
            Phi_phi0 = Uniform(minimum=0.0, maximum=2 * np.pi, name="Phi_phi0", boundary="periodic")
            Phi_r0 = Uniform(minimum=0.0, maximum=2 * np.pi, name="Phi_r0", boundary="periodic")
            xI0 = DeltaFunction(peak=1.0, name="xI0")
            Phi_theta0 = DeltaFunction(peak=0.0, name="Phi_theta0")
        """
        return LISAPriorDict.from_string(emri_prior_text)
    
    @classmethod
    def default(cls, use_cupy: bool = False, return_gpu: bool = False) -> "EMRIConfig":
        return cls(None, use_cupy, return_gpu)
    

class HyperConfig(BaseSourceConfig):
    """Configuration for LISA PSD parameters."""

    def __init__(
        self, 
        prior_file: str | None = None, 
        use_cupy: bool = False, 
        return_gpu: bool = False
    ):
        prior_dict = (
            LISAPriorDict.from_file(prior_file) 
            if prior_file is not None 
            else self._get_default_prior_dict()
        )
        super().__init__(
            prior_dict=prior_dict,
            source_name="hyper",
            use_cupy=use_cupy,
            return_gpu=return_gpu
        )

    @classmethod
    def _get_default_prior_dict(cls) -> "LISAPriorDict":
        hyper_prior_text = """
            model = Categorical(n_categories=2, name="model", name_phys="model", latex_label=r"$M$")
        """
        return LISAPriorDict.from_string(hyper_prior_text)

    @classmethod
    def default(cls, use_cupy: bool = False, return_gpu: bool = False) -> "HyperConfig":
        return cls(None, use_cupy, return_gpu)


from .network import HyperGalaxyPrior
from .base import UniformDistribution
from .discrete import HyperPoisson
from .analytical import CosineUniform, DeltaFunction, ResolvabilityPrior
import numpy as np


class HyperGBConfig(BaseSourceConfig):
    """
    Complete configuration class for Hyper-Parameter Population Inference of Galactic Binaries.
    Exports three branches to Eryn:
        - 'gb': The parameter prior (NF + Geometric Uniforms).
        - 'num_gbs': The population counts (Poisson).
        - 'resolv_gb': The detection probability (erf SNR).
    """

    def __init__(
        self,
        nf_config_files: list[str],
        poisson_lams: list[float],
        rho_threshold: float = 7.0,
        sigma_resolv: float = 1.0,
        use_cupy: bool = False,
        return_gpu: bool = False,
        support_floor: float | None = None,
    ):
        """
        Args:
            support_floor: Mixture weight of R5 method 2, or ``None`` (the default) to
                leave the flows exactly as they were trained. See
                :class:`lisatools.globalfit.priors.network.SupportFloor`: switching it
                on changes the target distribution, so it is a modelling decision.
        """
        if len(nf_config_files) != len(poisson_lams):
            raise ValueError("The number of NF config files must match the number of Poisson lambdas.")

        self.use_cupy = use_cupy
        self.return_gpu = return_gpu
        self.support_floor = support_floor

        self.gb_priors = LISAPriorDict({
            ("logA", "f0_mHz", "fdot", "ra", "sin_dec"): HyperGalaxyPrior(
                config_files=nf_config_files,
                use_cupy=use_cupy,
                return_gpu=return_gpu,
                support_floor=support_floor,
            ),
            "phi0": UniformDistribution(minimum=0.0, maximum=2*np.pi, name="phi0", boundary="periodic"),
            "cos_inc": CosineUniform(name="cos_inc", name_phys="inc"),
            "psi": UniformDistribution(minimum=0.0, maximum=np.pi, name="psi", boundary="periodic"),
            "fddot": DeltaFunction(peak=0.0, name="fddot")
        })

        super().__init__(self.gb_priors, "gb", use_cupy, return_gpu)
        
        output_order = ["logA", "f0_mHz", "fdot", "phi0", "cos_inc", "psi", "ra", "sin_dec"]
        self.priors["gb"].reset_key_order(output_order)

        self.transform = TransformContainer(
            input_basis=output_order,
            output_basis=['A', 'f0', 'fdot', 'fddot', 'phi0', 'inc', 'psi', 'ra', 'dec'],
            parameter_transforms=self.transform.original_parameter_transforms,
            fill_dict={'fddot': 0.0},
            key_map={'logA': 'A', 'f0_mHz': 'f0', 'sin_dec': 'dec', 'cos_inc': 'inc'},
        )
        
        self.periodic = PeriodicContainer(
            periodic={self.source_name: {'phi0': 2*np.pi, 'psi': np.pi, 'ra': 2*np.pi}},
            key_order={self.source_name: tuple(self.transform.input_basis)}
        )
        
        self.num_gbs_prior = HyperPoisson(
            lams=poisson_lams,
            use_cupy=use_cupy,
            return_gpu=return_gpu
        )

        self.resolv_gb_prior = ResolvabilityPrior(
            rho_threshold=rho_threshold,
            sigma=sigma_resolv,    
            use_cupy=use_cupy,
            return_gpu=return_gpu
        )

    def get_multi_branch_priors(self) -> dict[str, ProbDistContainer]:
        """Returns the multi-branch dictionary expected by Eryn's global state."""
        return {
            "gb": self.priors["gb"],
            "num_gbs": ProbDistContainer({0: self.num_gbs_prior}, use_cupy=self.use_cupy, return_gpu=self.return_gpu),
            "resolv_gb": ProbDistContainer({0: self.resolv_gb_prior}, use_cupy=self.use_cupy, return_gpu=self.return_gpu)
        }
        

from .network import HyperGalForPrior

class HyperGalForConfig(BaseSourceConfig):
    """
    Complete configuration class for Hyper-Parameter Inference of the Galactic Foreground.
    Exports the 'galfor' branch to Eryn.
    """

    def __init__(
        self,
        nf_config_files: list[str],
        use_cupy: bool = False,
        return_gpu: bool = False,
        support_floor: float | None = None,
    ):
        """
        Args:
            support_floor: Mixture weight of R5 method 2, or ``None`` (the default) to
                leave the flows exactly as they were trained. This is the density
                defect B9 lives in: without a floor the two models' foreground fits
                assign each other's states zero density, so the model move can never
                accept. See
                :class:`lisatools.globalfit.priors.network.SupportFloor`.
        """
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu
        self.support_floor = support_floor

        # Initialize the Prior dictionary. All 5 parameters are handled by the HyperPrior.
        self.galfor_priors = LISAPriorDict({
            ("log10_Amp", "alpha", "log10_f1", "log10_fknee", "log10_f2"): HyperGalForPrior(
                config_files=nf_config_files,
                use_cupy=use_cupy,
                return_gpu=return_gpu,
                support_floor=support_floor,
            )
        })

        super().__init__(self.galfor_priors, "galfor", use_cupy, return_gpu)

