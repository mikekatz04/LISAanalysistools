import numpy as np
from typing import Sequence, Callable, Tuple, Any

from numpy.typing import ArrayLike
from ...utils.typing import NDArrayLike

from lisaflow.experiments.rvs.galaxy import Galaxy
from .joint import JointPrior

try:
    import torch
except ImportError:
    torch = None


class NormalizingFlowPrior(JointPrior):
    """
    Base class for Neural Network / Normalizing Flow priors.
    
    Seamlessly handles PyTorch <-> CuPy/NumPy conversion, batching to prevent 
    CUDA OOM, and min-max normalization mapping.
    """

    def __init__(
        self,
        names: Sequence[str],
        flow_model: Any, 
        param_min: ArrayLike,
        param_max: ArrayLike,
        device: str = "cpu",
        batch_size: int = 100_000,
        latex_labels: Sequence[str | None] | None = None,
        units: Sequence[str | None] | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if torch is None:
            raise ImportError("PyTorch must be installed to use NormalizingFlowPrior.")

        super().__init__(
            names=names,
            latex_labels=latex_labels,
            units=units,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )

        self.flow = flow_model
        self.dev = torch.device(device)
        self.flow.to(self.dev)
        self.flow.eval()  # Always set to evaluation mode for priors

        self.batch_size = batch_size

        # Register normalizations on the target PyTorch device
        self.param_min = torch.tensor(param_min, dtype=torch.float32, device=self.dev)
        self.param_max = torch.tensor(param_max, dtype=torch.float32, device=self.dev)
        
        #! handled in lisaflow
        # # log Jacobian of the [-1, 1] normalization transformation:
        # # y = 2 * (x - min) / (max - min) - 1  -->  dy/dx = 2 / (max - min)
        # # log |dy/dx| = sum( log(2) - log(max - min) )
        # self._norm_log_jacobian = torch.sum(
        #     np.log(2.0) - torch.log(self.param_max - self.param_min)
        # )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map inputs from [min, max] to [-1, 1]."""
        return 2.0 * (x - self.param_min) / (self.param_max - self.param_min) - 1.0

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Map inputs from [-1, 1] back to [min, max]."""
        return self.param_min + (x_norm + 1.0) * (self.param_max - self.param_min) / 2.0

    def rvs(self, size: int | Tuple[int, ...] = (1,), **kwargs: Any) -> NDArrayLike:
        """Sample from the flow and denormalize back to the target parameter space."""
        if isinstance(size, int):
            size = (size,)
        
        num_samples = int(np.prod(size))
        samples_list = []

        with torch.no_grad():
            remaining = num_samples
            while remaining > 0:
                current_batch = min(remaining, self.batch_size)
                # Sample in latent space [-1, 1]
                batch_samples_norm = self.flow.sample(current_batch)
                # Denormalize to trained space
                batch_samples = self._denormalize(batch_samples_norm)
                samples_list.append(batch_samples)
                remaining -= current_batch

        # Concatenate and reshape
        all_samples = torch.cat(samples_list, dim=0).reshape(*size, self.num_vars)
        
        # Move to CuPy/NumPy via DLPack or CPU numpy conversion
        if self.use_cupy and self.dev.type == "cuda":
            import cupy as cp
            from torch.utils.dlpack import to_dlpack
            out_array = cp.from_dlpack(to_dlpack(all_samples))
        else:
            out_array = self.xp.asarray(all_samples.cpu().numpy())

        return self._to_device(out_array)

    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        """Calculate log probability, properly accounting for min-max normalization."""
        # Ensure we have a PyTorch tensor on the correct device
        if self.use_cupy and hasattr(x, "toDlpack"):
            from torch.utils.dlpack import from_dlpack
            x_torch = from_dlpack(x.toDlpack()).to(self.dev, dtype=torch.float32)
        else:
            x_torch = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

        original_shape = x_torch.shape
        x_flat = x_torch.view(-1, self.num_vars)
        
        log_prob_flat = torch.zeros(x_flat.shape[0], device=self.dev)

        with torch.no_grad():
            # Process in batches to avoid OOM
            for start_idx in range(0, x_flat.shape[0], self.batch_size):
                end_idx = min(start_idx + self.batch_size, x_flat.shape[0])
                
                # normalize batch to [-1, 1]
                batch_norm = self._normalize(x_flat[start_idx:end_idx])
                
                # evaluate flow
                batch_log_prob = self.flow.log_prob(batch_norm)
                
                # apply Jacobian of normalization mapping
                log_prob_flat[start_idx:end_idx] = batch_log_prob + self._norm_log_jacobian

        # reshape to original batch dimensions
        log_prob = log_prob_flat.view(original_shape[:-1])

        if self.use_cupy and self.dev.type == "cuda":
            import cupy as cp
            from torch.utils.dlpack import to_dlpack
            out_array = cp.from_dlpack(to_dlpack(log_prob))
        else:
            out_array = self.xp.asarray(log_prob.cpu().numpy())

        return self._to_device(out_array)




class FullGalaxyPrior(JointPrior, Galaxy):
    """
    5D Normalizing Flow prior for Galactic Binaries.
    
    Network Fit Basis (NFB): [logA, logf0_Hz, -sign(fdot)*log|fdot|, ra, sin_dec]
    Sampling Basis (SB):     [logA, f0_mHz, fdot, ra, sin_dec]
    Physical Basis (PB):     [A, f0_Hz, fdot, ra, dec]
    """
    
    def __init__(
        self, 
        config_file: str, 
        use_cupy: bool = False,
        return_gpu: bool = False
    ):

        FullGalaxyPrior.__init__(self, config_file)
        self.load_fit()

        param_names = ("logA", "f0_mHz", "fdot", "ra", "sin_dec")
        param_names_phys = ("A", "f0", "fdot", "ra", "dec")
        latex_labels = (r"\log \mathcal{A}", r"f_{0, {\rm mHz}}", r"\dot{f}", r"\alpha", r"\sin \delta")

        JointPrior.__init__(
            self,
            names=param_names,
            names_phys=param_names_phys,
            latex_labels=latex_labels,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            inverse_transform_nd=self._sb_to_pb  # sampling to physical
        )

    def _sb_to_pb(self, logA, f0_mHz, fdot, ra, sin_dec, **kwargs):
        """Maps: Sampling Basis (SB) -> Physical Basis (PB)"""
        A = self.xp.exp(logA)
        f0 = f0_mHz * 1e-3
        dec = self.xp.arcsin(sin_dec)
        return A, f0, fdot, ra, dec

    def _nfb_to_sb(self, nfb_samples):
        """Maps: Network Fit Basis (NFB) -> Sampling Basis (SB)"""
        logA = nfb_samples[..., 0]
        logf0 = nfb_samples[..., 1]
        mlogfdot = nfb_samples[..., 2]
        ra = nfb_samples[..., 3]
        sin_dec = nfb_samples[..., 4]

        f0_mHz = self.xp.exp(logf0) * 1e3
        
        # y = -sign(fdot) * log|fdot|
        # Since log|fdot| is negative (fdot ~ 1e-16), |y| = -log|fdot|.
        # Thus |fdot| = exp(-|y|) and sign of y matches sign of fdot.
        fdot = self.xp.sign(mlogfdot) * self.xp.exp(-self.xp.abs(mlogfdot))

        return self.xp.stack([logA, f0_mHz, fdot, ra, sin_dec], axis=-1)

    def _sb_to_nfb(self, sb_samples):
        """Maps: Sampling Basis (SB) -> Network Fit Basis (NFB)"""
        logA = sb_samples[..., 0]
        f0_mHz = sb_samples[..., 1]
        fdot = sb_samples[..., 2]
        ra = sb_samples[..., 3]
        sin_dec = sb_samples[..., 4]

        # f0_mHz -> log(f0_Hz)
        logf0 = self.xp.log(f0_mHz * 1e-3)
        
        # fdot -> -sign(fdot) * log|fdot|
        # Safe guard against absolute 0 to prevent log(0) = -inf causing NaN gradients
        safe_fdot = self.xp.where(fdot != 0, fdot, 1e-30)
        mlogfdot = -self.xp.sign(safe_fdot) * self.xp.log(self.xp.abs(safe_fdot))
        return self.xp.stack([logA, logf0, mlogfdot, ra, sin_dec], axis=-1)

    def rvs(self, size: int | Tuple[int, ...] = (1,), **kwargs: Any) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
            
        num_samples = int(np.prod(size))
        
        nfb_samples = self.sample(num_samples=num_samples)
        nfb_samples = nfb_samples.reshape(*size, self.num_vars)
        
        sb_samples = self._nfb_to_sb(nfb_samples)
        
        return self._to_device(sb_samples)
    
    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        x_sb = self.xp.asarray(x)
        original_shape = x_sb.shape
        
        x_nfb = self._sb_to_nfb(x_sb)
        x_nfb_flat = x_nfb.reshape(-1, self.num_vars)
        
        logpdf_nfb_flat = self.log_prob(x_nfb_flat)
        logpdf_nfb = logpdf_nfb_flat.reshape(original_shape[:-1])
        
        # d(NFB_f0) / d(SB_f0) = d(ln(f0_mHz*1e-3)) / d(f0_mHz) = 1 / f0_mHz
        # d(NFB_fdot) / d(SB_fdot) = d(-sign(fdot)*ln|fdot|) / d(fdot) = 1 / |fdot|
        # Log Determinant = -ln(f0_mHz) - ln(|fdot|)
        f0_mHz = x_sb[..., 1]
        fdot = x_sb[..., 2]
        
        safe_fdot = self.xp.where(fdot != 0, self.xp.abs(fdot), 1e-30)
        
        log_jacobian = -self.xp.log(f0_mHz) - self.xp.log(safe_fdot)
        
        # Final probability in the Sampling Basis
        logpdf_sb = logpdf_nfb + log_jacobian
        
        return self._to_device(logpdf_sb)


class HyperGalaxyPrior(JointPrior):
    """
    Conditional Normalizing Flow Prior for Model Selection.
    
    Evaluates p(theta | M_i) by routing the parameter batch `x` to the 
    corresponding Normalizing Flow network based on `model_index`.
    """

    def __init__(
        self,
        config_files: Sequence[str],  # One config file per model
        use_cupy: bool = False,
        return_gpu: bool = False,
    ):
        self.n_models = len(config_files)
        
        # Initialize the list of underlying FullGalaxyPriors
        self.models = [
            FullGalaxyPrior(
                config_file=cfg, 
                use_cupy=use_cupy, 
                return_gpu=True, # Keep strictly on GPU internally
            ) for cfg in config_files
        ]

        names = self.models[0].names
        names_phys = self.models[0].names_phys
        latex_labels = self.models[0].latex_labels

        super().__init__(
            names=names,
            names_phys=names_phys,
            latex_labels=latex_labels,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            inverse_transform_nd=self.models[0].inverse_transform_nd
        )

    def logpdf(self, x: ArrayLike, model_index: ArrayLike, **kwargs: Any) -> NDArrayLike:
        """
        Evaluate p(x | M_i).
        
        Args:
            x: GB parameters. Shape (nwalkers, nleaves, ndim).
            model_index: Model indicators. Shape (nwalkers,) or (nwalkers, 1).
        """
        x_sb = self.xp.asarray(x)
        mod_idx = self.xp.asarray(model_index).astype(self.xp.int32)
        
        if mod_idx.ndim < x_sb.ndim - 1:
            mod_idx = self.xp.expand_dims(mod_idx, tuple(range(mod_idx.ndim, x_sb.ndim - 1)))
        
        mod_idx = self.xp.broadcast_to(mod_idx, x_sb.shape[:-1])
        
        out = self.xp.full(x_sb.shape[:-1], -np.inf, dtype=self.xp.float64)

        for i in range(self.n_models):
            mask = (mod_idx == i)
            
            if not mask.any():
                continue
            
            x_model_subset = x_sb[mask]
            logp_subset = self.models[i].logpdf(x_model_subset)
            out[mask] = logp_subset

        return self._to_device(out)

    def rvs(self, size: int | Tuple[int, ...] = (1,), model_index: ArrayLike = 0, **kwargs: Any) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        
        mod_idx = self.xp.asarray(model_index).astype(self.xp.int32)
        
        if mod_idx.shape != size:
            mod_idx = self.xp.broadcast_to(mod_idx, size)
            
        out_samples = self.xp.empty((*size, self.num_vars), dtype=self.xp.float64)
        
        for i in range(self.n_models):
            mask = (mod_idx == i)
            n_samples_model = int(mask.sum())
            if n_samples_model == 0:
                continue
                
            out_samples[mask] = self.models[i].rvs(size=n_samples_model)

        return self._to_device(out_samples)