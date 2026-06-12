from __future__ import annotations

import os
import ast
import numpy as np
from typing import Dict, Union, Tuple

# Assuming all prior classes are defined in the same package/module
# Adjust the import path according to your project structure
from .base import *
from .joint import *
from .analytical import *
from .network import *
from .discrete import *


class LISAPriorDict(dict):
    """
    A dictionary container for LISA priors.
    
    Provides secure file I/O for text-based prior configurations (similar to Bilby),
    allowing priors to be serialized to metadata files and safely reloaded.
    """

    @classmethod
    def from_string(cls, text_block: str) -> "LISAPriorDict":
        """
        Safely parses a multiline string of prior definitions into Prior objects.
        
        Args:
            text_block (str): The multiline string containing the prior configuration.
            
        Returns:
            LISAPriorDict: The populated dictionary of prior objects.
            
        Raises:
            ValueError: If a line cannot be parsed or evaluated safely.
        """
        safe_namespace = {
            "np": np, # for access to numpy functions/constants in prior definitions
            "Prior": Prior,
            "JointPrior": JointPrior,
            "DeltaFunction": DeltaFunction,
            "Uniform": UniformDistribution,
            "PowerLaw": PowerLaw,
            "LogUniform": LogUniform,
            "Log10Uniform": Log10Uniform,
            "CosineUniform": CosineUniform,
            "SineUniform": SineUniform,
            "Gaussian": Gaussian,
            "Normal": Normal,
            "LogNormal": LogNormal,
            "LogGaussian": LogGaussian,
            "Exponential": Exponential,
            "MultivariateGaussian": MultivariateGaussian,
            "MojitoF0FdotPrior": MojitoF0FdotPrior,
            "MojitoF0mHzFdotPrior": MojitoF0mHzFdotPrior,
            "DiscreteUniform": DiscreteUniform,
            "Categorical": Categorical,
            "Poisson": Poisson,
        }
        
        safe_env = {"__builtins__": {}}
        safe_env.update(safe_namespace)

        prior_dict = {}
        for line_num, line in enumerate(text_block.strip().split("\n"), start=1):
            line = line.strip()
            
            # Ignore empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            try:
                # Split on the first '=' sign
                key_str, val_str = line.split("=", 1)
                key_str = key_str.strip()
                val_str = val_str.strip()

                # Handle JointPriors which have tuple keys: e.g., ('A', 'f0') = ...
                if key_str.startswith("("):
                    key = ast.literal_eval(key_str)
                else:
                    # Remove quotes if the user put them around a single string key
                    key = key_str.strip("'\" ")

                # Evaluate the right-hand side using the heavily restricted namespace
                prior_obj = eval(val_str, safe_env)
                prior_dict[key] = prior_obj
                
            except Exception as e:
                raise ValueError(
                    f"Failed to parse line {line_num}: '{line}'. Error: {e}"
                )

        return cls(prior_dict)

    @classmethod
    def from_file(cls, filepath: str) -> "LISAPriorDict":
        """
        Reads a .prior text file and safely parses it into objects.
        
        Args:
            filepath (str): Path to the .prior text file.
            
        Returns:
            LISAPriorDict: The populated dictionary of prior objects.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Prior file {filepath} not found.")

        with open(filepath, "r") as f:
            return cls.from_string(f.read())

    def to_file(self, filepath: str) -> None:
        """
        Writes the dictionary to a clean, metadata-friendly text file.
        Uses the __repr__ method of the Prior objects to ensure reproducibility.
        
        Args:
            filepath (str): Path where the file should be saved.
        """
        with open(filepath, "w") as f:
            f.write("# LISA Prior Configuration File\n")
            f.write("# Lines beginning with '#' are ignored.\n\n")
            
            for key, prior in self.items():
                # If key is a tuple (JointPrior), format it cleanly
                if isinstance(key, tuple):
                    key_str = f"({', '.join(f'{repr(k)}' for k in key)})"
                else:
                    # Standard string key
                    key_str = str(key)
                    
                f.write(f"{key_str} = {repr(prior)}\n")

    def to_string(self) -> str:
        """
        Returns the text-file representation of the priors as a string.
        Useful for saving directly to HDF5 attributes (e.g., f.attrs["priors"]).
        """
        lines = ["# LISA Prior Configuration File\n"]
        for key, prior in self.items():
            if isinstance(key, tuple):
                key_str = f"({', '.join(f'{repr(k)}' for k in key)})"
            else:
                key_str = str(key)
            lines.append(f"{key_str} = {repr(prior)}")
            
        return "\n".join(lines)