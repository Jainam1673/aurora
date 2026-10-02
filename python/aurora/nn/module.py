"""Base Module class for AURORA neural network architectures."""

from __future__ import annotations

from typing import Any

from aurora.nn.parameter import Parameter


class Module:
    """Base class for all neural network modules."""

    def __init__(self) -> None:
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}
        self.training: bool = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if "_parameters" not in self.__dict__:
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if "_modules" not in self.__dict__:
                super().__setattr__("_modules", {})
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self, recurse: bool = True) -> list[Parameter]:
        """Return a list of module parameters."""
        params: list[Parameter] = list(self._parameters.values())
        if recurse:
            for mod in self._modules.values():
                params.extend(mod.parameters(recurse=True))
        return params

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> list[tuple[str, Parameter]]:
        """Return an iterator over module parameters, yielding name and parameter."""
        res: list[tuple[str, Parameter]] = []
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            res.append((full_name, param))
        if recurse:
            for mod_name, mod in self._modules.items():
                sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                res.extend(mod.named_parameters(prefix=sub_prefix, recurse=True))
        return res

    def modules(self) -> list[Module]:
        """Return all submodules in this module."""
        mods = [self]
        for m in self._modules.values():
            mods.extend(m.modules())
        return mods

    def train(self, mode: bool = True) -> Module:
        """Set module in training mode."""
        self.training = mode
        for mod in self._modules.values():
            mod.train(mode)
        return self

    def eval(self) -> Module:
        """Set module in evaluation mode."""
        return self.train(False)

    def zero_grad(self) -> None:
        """Zero gradients across all parameters."""
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Return a dictionary containing module parameter values and shapes."""
        out: dict[str, dict[str, Any]] = {}
        for name, param in self.named_parameters():
            out[name] = {
                "shape": list(param.shape),
                "data": [float(x) for x in param.numpy().flatten()],
            }
        return out

    def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
        """Copy parameters from state_dict into this module."""
        current_params = dict(self.named_parameters())
        for name, entry in state_dict.items():
            if name not in current_params:
                raise KeyError(f"Unexpected key '{name}' in state_dict")
            param = current_params[name]
            shape = tuple(entry["shape"])
            if param.shape != shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': module has {param.shape}, state_dict has {shape}"
                )
            import numpy as np

            param.data = np.array(entry["data"], dtype=param.dtype).reshape(shape)
