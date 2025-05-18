import typing
from typing import Any, Callable, List, Tuple, Union, Optional, cast, Dict, overload
import warnings
import torch
import numpy as np
from torch import Tensor
import functools

from captum._utils.gradient import _forward_layer_eval, _run_forward
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _extract_device,
)
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
    ModuleOrModuleList
)
from torch.nn import Module
from captum.log import log_usage
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
    _tensorize_baseline,
)
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from attribution_methods import run_from_layer_fn

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name


# MODIFICATIONS TO CAPTUM

def custom_compute_gradients(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
    get_outputs = False,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        # _run_forward may return future of Tensor,
        # but we don't support it here now
        # And it will fail before here.
        outputs = cast(Tensor, outputs)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(outputs), inputs)
    if get_outputs:
        return grads, outputs
    return grads


def custom_batch_attribution(
    attr_method,
    num_examples,
    internal_batch_size,
    n_steps: int,
    include_endpoint: bool = False,
    **kwargs: Any,
):
    if internal_batch_size < num_examples:
        warnings.warn(
            "Internal batch size cannot be less than the number of input examples. "
            "Defaulting to internal batch size of %d equal to the number of examples."
            % num_examples,
            stacklevel=1,
        )
    # Number of steps for each batch
    step_count = max(1, internal_batch_size // num_examples)
    if include_endpoint:
        if step_count < 2:
            step_count = 2
            warnings.warn(
                "This method computes finite differences between evaluations at "
                "consecutive steps, so internal batch size must be at least twice "
                "the number of examples. Defaulting to internal batch size of %d"
                " equal to twice the number of examples." % (2 * num_examples),
                stacklevel=1,
            )

    total_attr = None
    cumulative_steps = 0
    step_sizes_func, alphas_func = approximation_parameters(kwargs["method"])
    full_step_sizes = step_sizes_func(n_steps)
    full_alphas = alphas_func(n_steps)

    while cumulative_steps < n_steps:
        start_step = cumulative_steps
        end_step = min(start_step + step_count, n_steps)
        batch_steps = end_step - start_step

        if include_endpoint:
            batch_steps -= 1

        step_sizes = full_step_sizes[start_step:end_step]
        alphas = full_alphas[start_step:end_step]
        current_attr = attr_method._attribute(
            **kwargs, n_steps=batch_steps, step_sizes_and_alphas=(step_sizes, alphas)
        )

        if total_attr is None:
            total_attr = current_attr
        else:
            if isinstance(total_attr, Tensor):
                total_attr = total_attr + current_attr.detach()
            elif isinstance(total_attr, tuple) and len(total_attr) == 4:
                # total_attr returns gradients, outputs, steps, scaled features
                total_attr = tuple(
                    torch.cat((prev_total, current))
                    for prev_total, current in zip(total_attr, current_attr)
                )
            else:
                total_attr = tuple(
                    current.detach() + prev_total
                    for current, prev_total in zip(current_attr, total_attr)
                )
        if include_endpoint and end_step < n_steps:
            cumulative_steps = end_step - 1
        else:
            cumulative_steps = end_step
    return total_attr


class SplitIntegratedGradients(GradientAttribution):

    def __init__(self, forward_func: Callable, multiply_by_inputs: bool = True) -> None:
        GradientAttribution.__init__(self, forward_func)
        self.gradient_func = custom_compute_gradients
        self._multiply_by_inputs = multiply_by_inputs


    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]:
        ...

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        inputs, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inputs, baselines, n_steps, method)

        # Perform attribution and return shaped results 
        if internal_batch_size is not None:
            num_examples = inputs[0].shape[0]
            attributions = custom_batch_attribution(
                self,
                num_examples,
                internal_batch_size,
                n_steps,
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
            )
        else:
            attributions = self._attribute(
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )

        return attributions

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
    ) -> Tuple[Tensor, ...]:
        # Retrieve step size and scaling factor for approximation method
        if step_sizes_and_alphas is None:
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # Interpolated inputs to calculate sensitivity for
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        # Handle additional arguments to the forward pass
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # Calculate the gradients for each interpolated inputs
        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads, outputs = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
            get_outputs=True,
        )

        # Multiply gradients by step size
        # calling contiguous to avoid `memory whole` problems
        steps = torch.tensor(step_sizes).float().view(n_steps, 1)
        scaled_grads = [
            grad.contiguous().view(n_steps, -1) * steps.to(grad.device)
            for grad in grads
        ]

        # # aggregates across all steps for each tensor in the input tuple
        # # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        if not self.multiplies_by_inputs:
            attributions = total_grads[0], outputs, steps, scaled_features_tpl[0]
        else:
            multiplied_grads = tuple(
                grad * (input - baseline)
                for grad, input, baseline in zip(total_grads, inputs, baselines)
            )
            attributions = multiplied_grads[0], outputs, steps, scaled_features_tpl[0]
        return attributions


    def has_convergence_delta(self) -> bool:
        return True
    
    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs
    
from captum.attr import LayerIntegratedGradients

# class SplitLayerIntegratedGradients(LayerIntegratedGradients):

#     def __init__(
#         self,
#         forward_func: Callable[..., Tensor],
#         layer: ModuleOrModuleList,
#         device_ids: Union[None, List[int]] = None,
#         multiply_by_inputs: bool = True,
#     ):
#         LayerIntegratedGradients.__init__(self, forward_func, layer, device_ids=device_ids, multiply_by_inputs=multiply_by_inputs)
#         self.ig = SplitIntegratedGradients(forward_func)


class SplitLayerIntegratedGradients(LayerAttribution, GradientAttribution):
    r"""
    Layer Integrated Gradients is a variant of Integrated Gradients that assigns
    an importance score to layer inputs or outputs, depending on whether we
    attribute to the former or to the latter one.

    Integrated Gradients is an axiomatic model interpretability algorithm that
    attributes / assigns an importance score to each input feature by approximating
    the integral of gradients of the model's output with respect to the inputs
    along the path (straight line) from given baselines / references to inputs.

    Baselines can be provided as input arguments to attribute method.
    To approximate the integral we can choose to use either a variant of
    Riemann sum or Gauss-Legendre quadrature rule.

    More details regarding the integrated gradients method can be found in the
    original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        layer: ModuleOrModuleList,
        device_ids: Union[None, List[int]] = None,
        multiply_by_inputs: bool = True,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or any
                        modification of it
            layer (ModuleOrModuleList): Layer or list of layers for which attributions
                        are computed. For each layer the output size of the attribute
                        matches this layer's input or output dimensions, depending on
                        whether we attribute to the inputs or outputs of the
                        layer, corresponding to the attribution of each neuron
                        in the input or output of this layer.

                        Please note that layers to attribute on cannot be
                        dependent on each other. That is, a subset of layers in
                        `layer` cannot produce the inputs for another layer.

                        For example, if your model is of a simple linked-list
                        based graph structure (think nn.Sequence), e.g. x -> l1
                        -> l2 -> l3 -> output. If you pass in any one of those
                        layers, you cannot pass in another due to the
                        dependence, e.g.  if you pass in l2 you cannot pass in
                        l1 or l3.

            device_ids (list[int]): Device ID list, necessary only if forward_func
                        applies a DataParallel model. This allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not necessary to provide this argument.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in,
                        then this type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of layer integrated gradients, if `multiply_by_inputs`
                        is set to True, final sensitivity scores are being multiplied by
                        layer activations for inputs - layer activations for baselines.

        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids=device_ids)
        GradientAttribution.__init__(self, forward_func)
        self.ig = SplitIntegratedGradients(forward_func, multiply_by_inputs)

        if isinstance(layer, list) and len(layer) > 1:
            warnings.warn(
                "Multiple layers provided. Please ensure that each layer is"
                "**not** solely dependent on the outputs of"
                "another layer. Please refer to the documentation for more"
                "detail.",
                stacklevel=2,
            )

    def _make_gradient_func(
        self,
        num_outputs_cumsum: Tensor,
        attribute_to_layer_input: bool,
        grad_kwargs: Optional[Dict[str, Any]],
    ) -> Callable[..., Tuple[Tensor, ...]]:

        def _gradient_func(
            forward_fn: Callable[..., Tensor],
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target_ind: TargetType = None,
            additional_forward_args: Optional[object] = None,
            get_outputs = False,
        ) -> Tuple[Tensor, ...]:
            if self.device_ids is None or len(self.device_ids) == 0:
                scattered_inputs = (inputs,)
            else:
                # scatter method does not have a precise enough return type in its
                # stub, so suppress the type warning.
                scattered_inputs = scatter(  # type:ignore
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Tensor, typing.Tuple[Tensor, ...]]`.
                    inputs,
                    target_gpus=self.device_ids,
                )

            scattered_inputs_dict: Dict[
                torch.device, Union[Tensor, Tuple[Tensor, ...]]
            ] = {
                scattered_input[0].device: scattered_input
                for scattered_input in scattered_inputs
            }

            with torch.autograd.set_grad_enabled(True):

                def layer_forward_hook(
                    module: Module,
                    hook_inputs: Union[Tensor, Tuple[Tensor, ...]],
                    hook_outputs: Union[None, Tensor, Tuple[Tensor, ...]] = None,
                    layer_idx: int = 0,
                ) -> Union[Tensor, Tuple[Tensor, ...]]:
                    device = _extract_device(module, hook_inputs, hook_outputs)
                    is_layer_tuple = (
                        isinstance(hook_outputs, tuple)
                        # hook_outputs is None if attribute_to_layer_input == True
                        if hook_outputs is not None
                        else isinstance(hook_inputs, tuple)
                    )

                    if is_layer_tuple:
                        return cast(
                            Union[Tensor, Tuple[Tensor, ...]],
                            scattered_inputs_dict[device][
                                num_outputs_cumsum[layer_idx] : num_outputs_cumsum[
                                    layer_idx + 1
                                ]
                            ],
                        )

                    return scattered_inputs_dict[device][num_outputs_cumsum[layer_idx]]

                hooks = []
                try:

                    layers = self.layer
                    if not isinstance(layers, list):
                        layers = [self.layer]

                    for layer_idx, layer in enumerate(layers):
                        hook = None
                        # TODO:
                        # Allow multiple attribute_to_layer_input flags for
                        # each layer, i.e. attribute_to_layer_input[layer_idx]
                        if attribute_to_layer_input:
                            hook = layer.register_forward_pre_hook(
                                functools.partial(
                                    layer_forward_hook, layer_idx=layer_idx
                                )
                            )
                        else:
                            hook = layer.register_forward_hook(
                                functools.partial(
                                    layer_forward_hook, layer_idx=layer_idx
                                )
                            )

                        hooks.append(hook)

                    # the inputs is an empty tuple
                    # coz it is prepended into additional_forward_args
                    output = _run_forward(
                        self.forward_func, (), target_ind, additional_forward_args
                    )
                finally:
                    for hook in hooks:
                        if hook is not None:
                            hook.remove()

                # _run_forward may return future of Tensor,
                # but we don't support it here now
                # And it will fail before here.
                output = cast(Tensor, output)
                assert output[0].numel() == 1, (
                    "Target not provided when necessary, cannot"
                    " take gradient with respect to multiple outputs."
                )
                # torch.unbind(forward_out) is a list of scalar tensor tuples and
                # contains batch_size * #steps elements
                grads = torch.autograd.grad(
                    torch.unbind(output), inputs, **grad_kwargs or {}
                )
            if get_outputs:
                return grads, output
            return grads

        return _gradient_func

    @overload
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType,
        target: TargetType,
        additional_forward_args: Optional[object],
        n_steps: int,
        method: str,
        internal_batch_size: Union[None, int],
        return_convergence_delta: Literal[False],
        attribute_to_layer_input: bool,
        grad_kwargs: Optional[Dict[str, Any]],
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]: ...

    @overload
    def attribute(  # type: ignore
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType,
        target: TargetType,
        additional_forward_args: Optional[object],
        n_steps: int,
        method: str,
        internal_batch_size: Union[None, int],
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool,
        grad_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tensor,
    ]: ...

    @overload
    # pyre-fixme[43]: This definition does not have the same decorators as the
    #  preceding overload(s).
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Tensor,
        ],
    ]: ...

    @log_usage()
    # pyre-fixme[43]: This definition does not have the same decorators as the
    #  preceding overload(s).
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Tensor,
        ],
    ]:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to layer inputs or outputs of the model, depending on whether
        `attribute_to_layer_input` is set to True or False, using the approach
        described above.

        In addition to that it also returns, if `return_convergence_delta` is
        set to True, integral approximation delta based on the completeness
        property of integrated gradients.

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define the starting point from which integral
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                            - either a tensor with matching dimensions to
                              corresponding tensor in the inputs' tuple
                              or the first dimension is one and the remaining
                              dimensions match with the corresponding
                              input tensor.
                            - or a scalar, corresponding to a tensor in the
                              inputs' tuple. This scalar value is broadcasted
                              for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.

                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.

                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_steps (int, optional): The number of steps used by the approximation
                        method. Default: 50.
            method (str, optional): Method for approximating the integral,
                        one of `riemann_right`, `riemann_left`, `riemann_middle`,
                        `riemann_trapezoid` or `gausslegendre`.
                        Default: `gausslegendre` if no method is provided.
            internal_batch_size (int, optional): Divides total #steps * #examples
                        data points into chunks of size at most internal_batch_size,
                        which are computed (forward / backward passes)
                        sequentially. internal_batch_size must be at least equal to
                        #examples.

                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain internal_batch_size / num_devices examples.
                        If internal_batch_size is None, then all evaluations are
                        processed in one batch.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.

                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
            grad_kwargs (Dict[str, Any], optional): Additional keyword
                        arguments for torch.autograd.grad.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                Integrated gradients with respect to `layer`'s inputs
                or outputs. Attributions will always be the same size and
                dimensionality as the input or output of the given layer,
                depending on whether we attribute to the inputs or outputs
                of the layer which is decided by the input flag
                `attribute_to_layer_input`.

                For a single layer, attributions are returned in a tuple if
                the layer inputs / outputs contain multiple tensors,
                otherwise a single tensor is returned.

                For multiple layers, attributions will always be
                returned as a list. Each element in this list will be
                equivalent to that of a single layer output, i.e. in the
                case that one layer, in the given layers, inputs / outputs
                multiple tensors: the corresponding output element will be
                a tuple of tensors. The ordering of the outputs will be
                the same order as the layers given in the constructor.

            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                The difference between the total approximated and true
                integrated gradients. This is computed using the property
                that the total sum of forward_func(inputs) -
                forward_func(baselines) must equal the total sum of the
                integrated gradient.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                examples in inputs.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> lig = LayerIntegratedGradients(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer integrated gradients for class 3.
            >>> # attribution size matches layer output, Nx12x32x32
            >>> attribution = lig.attribute(input, target=3)
        """
        inps, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inps, baselines, n_steps, method)

        baselines = _tensorize_baseline(inps, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        def flatten_tuple(tup: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
            return tuple(
                cast(
                    List[Tensor],
                    sum(
                        (
                            (
                                list(x)
                                if isinstance(x, (tuple, list))
                                else cast(List[Tensor], [x])
                            )
                            for x in tup
                        ),
                        [],
                    ),
                )
            )

        if self.device_ids is None:
            self.device_ids = getattr(self.forward_func, "device_ids", None)

        inputs_layer = _forward_layer_eval(
            self.forward_func,
            inps,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        input_layer_list: List[Tuple[Tensor, ...]]
        # if we have one output
        if not isinstance(self.layer, list):
            input_layer_list = [cast(Tuple[Tensor, ...], inputs_layer)]
        else:
            input_layer_list = inputs_layer

        num_outputs = [1 if isinstance(x, Tensor) else len(x) for x in input_layer_list]
        num_outputs_cumsum = torch.cumsum(
            torch.IntTensor([0] + num_outputs), dim=0  # type: ignore
        )
        inputs_layer = flatten_tuple(input_layer_list)

        baselines_layer = _forward_layer_eval(
            self.forward_func,
            baselines,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        baselines_layer = flatten_tuple(baselines_layer)

        # inputs -> these inputs are scaled

        self.ig.gradient_func = self._make_gradient_func(
            num_outputs_cumsum, attribute_to_layer_input, grad_kwargs
        )

        all_inputs = (
            (inps + additional_forward_args)
            if additional_forward_args is not None
            else inps
        )

        attributions = self.ig.attribute.__wrapped__(  # type: ignore
            self.ig,  # self
            inputs_layer,
            baselines=baselines_layer,
            target=target,
            additional_forward_args=all_inputs,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=False,
        )

        if return_convergence_delta:
            start_point, end_point = baselines, inps
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return attributions, delta
        
        return attributions
    

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self) -> bool:
        return self.ig.multiplies_by_inputs
    

# SPLIT INTEGRATED GRADIENTS IMPLEMENTATION

def reshape_fortran(data, shape):
    device = data.device
    data = np.reshape(data.detach().cpu().numpy(), shape, order='F')
    return torch.tensor(data).to(device)


def compute_layer_to_output_attributions_split_ig(
    model: HookedTransformer,
    original_input, 
    layer_input, 
    layer_baseline, 
    target_layer, 
    prev_layer, 
    metric, 
    metric_labels, 
    ratio=1.0,
):
    n_steps = 50
    n_samples = original_input.size(0)

    # Take the model starting from the target layer
    forward_fn = lambda x: run_from_layer_fn(model, original_input, prev_layer, x, metric, metric_labels)

    # Attribute to the target_layer's output
    split_ig = SplitLayerIntegratedGradients(forward_fn, target_layer, multiply_by_inputs=True)
    attributions = split_ig.attribute(inputs=layer_input,
                                    baselines=layer_baseline,
                                    n_steps=n_steps,
                                    internal_batch_size=n_samples, # Needs to match patching shape
                                    attribute_to_layer_input=False,
                                    return_convergence_delta=False)
    grads, outputs, _, _ = attributions

    # [n_samples, n_steps, seq_len, d_heads, d_model]
    grads_shape = (n_samples, n_steps,) + grads.shape[1:]
    grads = reshape_fortran(grads, grads_shape)
    outputs = reshape_fortran(outputs, (n_samples, n_steps))

    # Sanity check: shape of output
    # plt.plot(outputs[0].detach().cpu())
    # plt.show()

    left_igs = torch.zeros((n_samples,) + grads.shape[2:])
    right_igs = torch.zeros((n_samples,)+ grads.shape[2:])
    threshold_indices = torch.zeros((n_samples,))

    for i in range(n_samples):
        sample_output = outputs[i] # Shape [n_steps]
        sample_grads = grads[i] # Shape [n_steps, seq_len, d_heads, d_model]

        # Calculate threshold for split integrated gradients
        min_out, min_index = torch.min(sample_output, dim=0)
        max_out, max_index = torch.max(sample_output, dim=0)
        # Take output values up until threshold
        threshold = min_out + ratio * (max_out - min_out)

        if ratio == 1:
            assert torch.isclose(threshold, max_out), f"Threshold {threshold} is not equal to maximum {max_out} at {max_index}, when minimum is {min_out} at {min_index}"

        if max_index < min_index:
            # Flip direction such that min_index < max_index
            sample_output = sample_output.flip(dims=(0,))
            sample_grads = sample_grads.flip(dims=(0,))

        mask = sample_output > threshold + 1e-5 # Handle float precision errors
        if torch.count_nonzero(mask) > 0:
            print(f"There are outputs above the threshold {threshold}. {mask}")
            threshold_index = mask.int().argmax(dim=0)
            print(f"Value at threshold index: {sample_output[threshold_index]}")
            if max_index < min_index:
                threshold_indices[i] = n_steps - threshold_index
            else:
                threshold_indices[i] = threshold_index
        else:
            # No value is above threshold, take full IG
            threshold_index = n_steps
            threshold_indices[i] = threshold_index

        if ratio == 1:
            assert threshold_indices[i] == n_steps, f"Threshold index is {threshold_indices[i]}, expected {n_steps}."

        # Grads have already been scaled and multiplied.
        left_igs[i] = sample_grads[:threshold_index].sum(dim=0)
        right_igs[i] = sample_grads[threshold_index:].sum(dim=0)

    return left_igs, right_igs, threshold_indices


def split_integrated_gradients(model: HookedTransformer, clean_tokens: torch.Tensor, clean_cache: ActivationCache, corrupted_cache: ActivationCache, metric: callable, metric_labels, ratio: float):
    n_samples = clean_tokens.size(0)
    
    # Gradient attribution for neurons in MLP layers
    mlp_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.d_mlp)
    # Gradient attribution for attention heads
    attn_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.n_heads)

    # Calculate integrated gradients for each layer
    for layer in range(model.cfg.n_layers):

        # Gradient attribution on heads
        hook_name = get_act_name("result", layer)
        target_layer = model.hook_dict[hook_name]
        prev_layer_hook = get_act_name("z", layer)
        prev_layer = model.hook_dict[prev_layer_hook]

        layer_clean_input = clean_cache[prev_layer_hook]
        layer_corrupt_input = corrupted_cache[prev_layer_hook]

        # Shape [batch, seq_len, d_head, d_model]
        left_ig, right_ig, _ = compute_layer_to_output_attributions_split_ig(
            model, clean_tokens, layer_corrupt_input, layer_clean_input, target_layer, prev_layer, metric, metric_labels, ratio)

        # Calculate attribution score based on mean over each embedding, for each token
        per_token_score = left_ig.mean(dim=3)
        score = per_token_score.mean(dim=1)
        attn_results[:, layer] = score

        # Gradient attribution on MLP neurons
        hook_name = get_act_name("post", layer)
        target_layer = model.hook_dict[hook_name]
        prev_layer_hook = get_act_name("mlp_in", layer)
        prev_layer = model.hook_dict[prev_layer_hook]

        layer_clean_input = clean_cache[prev_layer_hook]
        layer_corrupt_input = corrupted_cache[prev_layer_hook]
        
        # Shape [batch, seq_len, d_model]
        left_ig, _, _ = compute_layer_to_output_attributions_split_ig(
            clean_tokens, layer_corrupt_input, layer_clean_input, target_layer, prev_layer, metric, metric_labels, ratio)
        score = left_ig.mean(dim=1)
        mlp_results[:, layer] = score

    return mlp_results, attn_results