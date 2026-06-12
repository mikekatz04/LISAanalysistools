from importlib import import_module
import json
import os
import re
from typing import Optional, Tuple

from lisatools.utils.typing import ArrayType 
from numpy.typing import ArrayLike, NDArray
import numpy as np
from scipy.interpolate import interp1d


class Prior(object):
    _default_latex_labels = {}

    def __init__(
        self, 
        name: Optional[str] = None, 
        latex_label: Optional[str] = None, 
        unit: Optional[str] = None, 
        minimum: float = -np.inf,
        maximum: float = np.inf, 
        check_range_nonzero: bool = True,
        use_cupy: bool = False,
        return_gpu: bool = False,
    ):
        """ Implements a Prior object

        Arg:
            name (str, optional): Name associated with prior.
            latex_label (str, optional): Latex label associated with prior, used for plotting.
            unit (str, optional): If given, a Latex string describing the units of the parameter.
            minimum (float, optional): Minimum of the domain, default=-np.inf
            maximum (float, optional): Maximum of the domain, default=np.inf
            check_range_nonzero (boolean, optional): If True, checks that the prior range is non-zero
            use_cupy (boolean, optional): If True, uses CuPy for GPU acceleration
            return_gpu (boolean, optional): If True, returns GPU arrays when use_cupy is True
        """
        if check_range_nonzero and maximum <= minimum:
            raise ValueError(
                f"maximum {maximum} <= minimum {minimum} for {type(self).__name__} prior on {name}"
                )

        self.name = name
        self.latex_label = latex_label
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.check_range_nonzero = check_range_nonzero
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu

    def rvs(
        self, 
        *args, 
        size: int | Tuple[int, ...] = (1,), 
        **kwargs
    ) -> ArrayType:
        """Draws a random variable(s) from the prior distribution

        Args:
            size (int | Tuple[int, ...], optional): Size of the sample. Defaults to (1,).

        Raises:
            ValueError: If size is not an integer or a tuple of integers.

        Returns:
            ArrayType: A random sample drawn from the prior distribution, with the specified size.
        """
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")
        
        if isinstance(size, int):
            size = (size,)
        
        raise NotImplementedError("rvs method is implemented in subclass")

    def pdf(
        self, 
        x: ArrayLike,
        *args,
        **kwargs
    ) -> ArrayType:
        """Calculates the probability density of the prior distribution at the given value(s).

        Args:
            x (ArrayLike): The value(s) for which to calculate the probability density.

        Returns:
            ArrayType: The probability density of the given value(s).
        """
        raise NotImplementedError("pdf method is implemented in subclass")

    def cdf(
        self, 
        x: ArrayLike,
        *args,
        **kwargs
    ) -> ArrayType:
        """ Generic method to calculate CDF, can be overwritten in subclass """
        from scipy.integrate import cumulative_trapezoid
        if np.any(np.isinf([self.minimum, self.maximum])):
            raise ValueError(
                "Unable to use the generic CDF calculation for priors with"
                "infinite support")
        x_arr = np.linspace(self.minimum, self.maximum, 1000)
        pdf = self.pdf(x)
        cdf = cumulative_trapezoid(pdf, x_arr, initial=0)
        interp = interp1d(x_arr, cdf, assume_sorted=True, bounds_error=False,
                          fill_value=(0, 1)) # type: ignore
        output = interp(x)
        if isinstance(x, (int, float)):
            output = float(output)
        return output

    def logpdf(
        self, 
        x: ArrayLike,
        *args,
        **kwargs
    ) -> ArrayType:
        """The log probability density of the prior distribution at the given value(s).

        Args:
            x (ArrayLike): The value(s) for which to calculate the log probability density.

        Returns:
            ArrayType: The log probability density of the given value(s).
        """
        raise NotImplementedError("logpdf method is implemented in subclass")

    def is_in_prior_range(self, x: ArrayLike):
        if self.minimum and self.maximum is not None:
            return (x >= self.minimum) & (x <= self.maximum)
        else:
            samples = self.rvs(size=1000)
            minimum, maximum = samples.min(), samples.max()
            return (x >= minimum) & (x <= maximum)

    @property
    def latex_label(self):
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label=None):
        if latex_label is None:
            self.__latex_label = self.__default_latex_label
        else:
            self.__latex_label = latex_label

    @property
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, unit):
        self.__unit = unit

    @property
    def latex_label_with_unit(self):
        """ If a unit is specified, returns a string of the latex label and unit """
        if self.unit is not None:
            return f"{self.latex_label} [{self.unit}]"
        else:
            return self.latex_label

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum

    @property
    def width(self):
        return self.maximum - self.minimum

    @property
    def __default_latex_label(self):
        if self.name in self._default_latex_labels.keys():
            label = self._default_latex_labels[self.name]
        else:
            label = self.name
        return label

    @classmethod
    def _parse_argument_string(cls, val):
        """
        Parse a string into the appropriate type for prior reading.

        Four tests are applied in the following order:

        - If the string is 'None':
            `None` is returned.
        - Else If the string is a raw string, e.g., r'foo':
            A stripped version of the string is returned, e.g., foo.
        - Else If the string contains ', e.g., 'foo':
            A stripped version of the string is returned, e.g., foo.
        - Else If the string contains an open parenthesis, (:
            The string is interpreted as a call to instantiate another prior
            class, Bilby will attempt to recursively construct that prior,
            e.g., Uniform(minimum=0, maximum=1), my.custom.PriorClass(**kwargs).
        - Else If the string contains a ".":
            It is treated as a path to a Python function and imported, e.g.,
            "some_module.some_function" returns
            :code:`import some_module; return some_module.some_function`
        - Else:
            Try to evaluate the string using `eval`. Only built-in functions
            and numpy methods can be used, e.g., np.pi / 2, 1.57.


        Parameters
        ==========
        val: str
            The string version of the argument

        Returns
        =======
        val: object
            The parsed version of the argument.

        Raises
        ======
        TypeError:
            If val cannot be parsed as described above.
        """
        if val == 'None':
            val = None
        elif re.sub(r'\'.*\'', '', val) in ['r', 'u']:
            val = val[2:-1]
        elif val.startswith("'") and val.endswith("'"):
            val = val.strip("'")
        elif '(' in val and not val.startswith(("[", "{")):
            other_cls = val.split('(')[0]
            vals = '('.join(val.split('(')[1:])[:-1]
            if "." in other_cls:
                module = '.'.join(other_cls.split('.')[:-1])
                other_cls = other_cls.split('.')[-1]
            else:
                module = __name__.replace('.' + os.path.basename(__file__).replace('.py', ''), '')
            other_cls = getattr(import_module(module), other_cls)
            val = other_cls.from_repr(vals)
        else:
            try:
                val = eval(val, dict(), dict(np=np, inf=np.inf, pi=np.pi))
            except NameError:
                if "." in val:
                    module = '.'.join(val.split('.')[:-1])
                    func = val.split('.')[-1]
                    new_val = getattr(import_module(module), func, val)
                    if val == new_val:
                        raise TypeError(
                            "Cannot evaluate prior, "
                            f"failed to parse argument {val}"
                        )
                    else:
                        val = new_val
        return val


class PriorException(Exception):
    """ General base class for all prior exceptions """