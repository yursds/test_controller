import torch

from tensordict.nn  import TensorDictModule
from torch          import nn

from collections    import defaultdict

import tqdm

from envPendulum import env

# Training a simple policy
#
# In this example, we will train a simple policy using the reward as a differentiable objective, 
# such as a negative loss. Our dynamic system is fully differentiable to backpropagate through 
# the trajectory return and adjust the weights of our policy to maximize this value directly.


# If GPU is available
mydevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen = torch.Generator(device=mydevice)
gen.manual_seed(0)


net = nn.Sequential(
    nn.LazyLinear(64, device=mydevice),
    nn.Tanh(),
    nn.LazyLinear(64, device=mydevice),
    nn.Tanh(),
    nn.LazyLinear(64, device=mydevice),
    nn.Tanh(),
    nn.LazyLinear(1, device=mydevice),
)
policy = TensorDictModule(
    net,
    in_keys=["observation"],
    out_keys=["action"],
)

optim = torch.optim.Adam(policy.parameters(), lr=2e-3)

# Training loop
batch_size = 32
iter = 20_000
roll = 100
pbar = tqdm.tqdm(range(iter // batch_size))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iter)
logs = defaultdict(list)

for _ in pbar:
    
    init_td = env.reset(env.gen_params(batch_size=[batch_size]))
    rollout = env.rollout(roll, policy, tensordict=init_td, auto_reset=False)
    traj_return = rollout["next", "reward"].mean()
    (-traj_return).backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    pbar.set_description(
        f"reward: {traj_return: 4.4f}, "
        f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
    )
    logs["return"].append(traj_return.item())
    logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
    scheduler.step()


from matplotlib import pyplot as plt

plt.ion()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(logs["return"])
plt.title("returns")
plt.xlabel("iteration")
plt.subplot(1, 2, 2)
plt.plot(logs["last_reward"])
plt.title("last reward")
plt.xlabel("iteration")

print('Complete')
plt.ioff()
plt.show()