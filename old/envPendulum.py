# ----------------------------------------------------------------------------------------------------------------------- #
# REFERENCE: https://github.com/pytorch/tutorials/blob/main/advanced_source/pendulum.py

# How to design an environment in TorchRL:
#   - Writing specs (input, observation and reward);
#   - Implementing behavior: seeding, reset and step.
# Transforming your environment:
#   - inputs, outputs, own transforms;
# How to use :class:`~tensordict.TensorDict`

# * `environments <https://pytorch.org/rl/reference/envs.html>`__
# * `transforms <https://pytorch.org/rl/reference/envs.html#transforms>`__
# * `models (policy and value function) <https://pytorch.org/rl/reference/modules.html>`__

# We will be designing a *stateless* environment.
# Stateless environments are more generic and hence cover a broader range of features of the environment API in TorchRL.

# The problem analized:
# A pendulum over which we can control the torque applied on its fixed point. Our goal is to place the pendulum in upward 
# position (angular position at 0 by convention) and having it standing still in that position. A reward used is 
# r = -(\theta^2 + 0.1 * \dot{\theta}^2 + 0.001 * u^2) which will be maximized when the angle is close to 0 (pendulum in 
# upward position), the angular velocity is close to 0 (no motion) and the torque is 0 too.
# ------------------------------------------------------------------------------------------------------------------------ #
 
import warnings

from typing         import Optional

import torch
from tensordict     import TensorDict, TensorDictBase

from torchrl.data   import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs   import(
                            CatTensors,
                            EnvBase,
                            Transform,
                            TransformedEnv,
                            UnsqueezeTransform,
                          )
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils                 import check_env_specs, step_mdp

warnings.filterwarnings("ignore") # gymnasium have some warning not corrected

DEFAULT_X = torch.pi
DEFAULT_Y = 1.0

# UTILS
def angle_normalize(x:torch.float16) -> torch.float16:
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

def make_composite_from_td(td:TensorDictBase) -> CompositeSpec:
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


class PendulumEnv(EnvBase):
    
    metadata:dict = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked:bool = False

    def __init__(self, device:str, td_params:TensorDictBase=None, seed:torch.float16=None):
        """ Initialize PendulumEnv and EnvBase class """
        super().__init__(device=device, batch_size=[])
        if td_params is None:
            td_params = self.gen_params()
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64, device=device).random_().item()
        self.set_seed(seed)

    def _step(self, tensordict:TensorDictBase)->TensorDictBase:
        """ `~torchrl.envs.EnvBase._step` expects a ``tensordict`` instance with an ``"action"`` key.
            1. Read the input keys (such as ``"action"``) and execute the simulation based on these;
            2. Retrieve observations, done state and reward;
            3. Write the set of observation values along with the reward and done state
            at the corresponding entries in a new :class:`TensorDict`."""
        th      = torch.Tensor(tensordict["th"])
        thdot   = torch.Tensor(tensordict["thdot"])
        g_force = torch.Tensor(tensordict["params", "g"])
        mass    = torch.Tensor(tensordict["params", "m"])
        length  = torch.Tensor(tensordict["params", "l"])
        dt      = torch.Tensor(tensordict["params", "dt"])
        
        u       = torch.Tensor(tensordict["action"]).squeeze(-1)
        u       = u.clamp(-tensordict["params", "max_torque"], tensordict["params", "max_torque"])
       
        costs   = torch.Tensor(angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2))

        new_thdot = torch.Tensor((thdot + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length**2) * u) * dt))
        new_thdot = (new_thdot.clamp(-tensordict["params", "max_speed"], tensordict["params", "max_speed"]))
        new_th = th + new_thdot * dt
        
        reward = -costs.view(*tensordict.shape, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)
        out = TensorDict(
            {
                "th": new_th,
                "thdot": new_thdot,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict:TensorDictBase)->TensorDictBase:
        """ `~torchrl.envs.EnvBase._reset` expects a ``tensordict`` as input.
            Like _step method, it should write the observation entries # and possibly a done state in the 
            ``tensordict`` it outputs.
            Important thing to consider is whether _reset method contains all the expected observations. """
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

        high_th = torch.tensor(DEFAULT_X, device=self.device)
        high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
        low_th = -high_th
        low_thdot = -high_thdot

        # for non batch-locked environments, the input ``tensordict`` shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        th = (torch.rand(tensordict.shape, generator=self.rng, device=self.device) * (high_th - low_th) + low_th)
        thdot = (torch.rand(tensordict.shape, generator=self.rng, device=self.device) * (high_thdot - low_thdot) + low_thdot)
        out = TensorDict(
            {
                "th": th,
                "thdot": thdot,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out
    
    def _make_spec(self, td_params:TensorDictBase) -> None:
        """ ``env.*_spec``
            The specs define the input and output domain of the environment. It is important that the specs 
            accurately define the tensors that will be received at runtime.
            * :obj:`EnvBase.observation_spec`: This will be a :class:`~torchrl.data.CompositeSpec` instance 
            where each key is an observation (`CompositeSpec` can be viewed as a dictionary of specs).
            * :obj:`EnvBase.action_spec`: It corresponds to the ``"action"`` entry in the input ``tensordict``.
            * :obj:`EnvBase.reward_spec`: provides information about the reward space;
            * :obj:`EnvBase.done_spec`: provides information about the space of the done flag.
            TorchRL specs are organized in two general containers: ``input_spec`` which
            contains the specs of the information that the step function reads (divided
            between ``action_spec`` containing the action and ``state_spec`` containing
            all the rest), and ``output_spec`` which encodes the specs that the
            step outputs (``observation_spec``, ``reward_spec`` and ``done_spec``).
            NOTE: In general, you should interact directly only with content: ``observation_spec``,
            ``reward_spec``, ``done_spec``, ``action_spec`` and ``state_spec``. """
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = CompositeSpec(
            th=BoundedTensorSpec(
                low=-torch.pi,
                high=torch.pi,
                shape=(),
                dtype=torch.float32,
                device = self.device
            ),
            thdot=BoundedTensorSpec(
                low=-td_params["params", "max_speed"],
                high=td_params["params", "max_speed"],
                shape=(),
                dtype=torch.float32,
                device = self.device
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = BoundedTensorSpec(
            low=-td_params["params", "max_torque"],
            high=td_params["params", "max_torque"],
            shape=(1,),
            dtype=torch.float32,
            device=self.device
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    def gen_params(self, batch_size=None) -> TensorDictBase:
        """ Define initial parameters """
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "max_speed": 8,
                        "max_torque": 2.0,
                        "dt": 0.05,
                        "g": 10.0,
                        "m": 1.0,
                        "l": 1.0,
                    },
                    [],
                )
            },
            [],
            device=self.device
        )
        if batch_size:
            td = td.expand(batch_size)
        return td

    def _set_seed(self, seed: Optional[int]) -> None:
        """ Reproducible experiments: seeding """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        self.rng = gen
        #rng = torch.manual_seed(seed)
        #self.rng = rng

        # Helpers: _make_step and gen_params
    
    
# If GPU is available
mydevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mydevice = torch.device("cpu")
env:PendulumEnv = PendulumEnv(device=mydevice)
check_env_specs(env)

class SinTransform(Transform):
    def _apply_transform(self, obs:torch.Tensor) -> torch.Tensor:
        return obs.sin()

    # The transform must also modify the data at reset time
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all in_keys/out_keys 
    # pairs and write the result in the observation_spec which is of type ``Composite``.
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec) -> BoundedTensorSpec:
        return BoundedTensorSpec(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )

class CosTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.cos()

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all in_keys/out_keys 
    # pairs and write the result in the observation_spec which is of type ``Composite``.
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )

# Transforming an environment.
envT:TransformedEnv = TransformedEnv(
    env,
    # We ``unsqueeze`` the entries ``["th", "thdot"]`` and pass them as ``in_keys_inv`` to squeeze them back.
    UnsqueezeTransform(
        unsqueeze_dim=-1,
        in_keys=["th", "thdot"],
        in_keys_inv=["th", "thdot"],
    ),
)

# Let us code new transforms that will compute the ``sine`` and ``cosine`` values of the position angle,
# as these values are more useful to us to learn a policy than the raw angle value: """
t_sin = SinTransform(in_keys=["th"], out_keys=["sin"])
t_cos = CosTransform(in_keys=["th"], out_keys=["cos"])

# Concatenates the observations onto an "observation" entry.
# ``del_keys=False`` ensures that we keep these values for the next iteration.
cat_transform = CatTensors(
    in_keys=["sin", "cos", "thdot"], 
    dim=-1, 
    out_key="observation", 
    del_keys=False
)

envT.append_transform(t_sin)
envT.append_transform(t_cos)
envT.append_transform(cat_transform)
env = envT

# Check
check_env_specs(env)

