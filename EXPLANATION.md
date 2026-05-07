# HW2 Policy Gradient Code Walkthrough

This document explains what each file in the project is doing and how the parts fit together in practice.

## Big Picture

The code implements a **policy gradient reinforcement learning loop**:

1. Collect experience from the environment using the current policy.
2. Compute returns / advantages from the collected rewards.
3. Update the policy (actor) to make good actions more likely.
4. Optionally update a value network (critic / baseline) to reduce variance.
5. Repeat.

A single training iteration usually looks like this:

```text
run.py
  └─ sample trajectories from the env
      └─ utils.py
          └─ returns lists of observations, actions, rewards, terminals
  └─ PGAgent.update(...)
      ├─ compute Q-values from rewards
      ├─ flatten trajectory lists into batch arrays
      ├─ compute advantages
      ├─ update policy network
      └─ update critic network (if enabled)
```

## File-by-file quick reference

| File | Main job | In plain words |
| --- | --- | --- |
| `src/scripts/run.py` | Starts training and controls the loop | "Run the experiment" |
| `src/infrastructure/utils.py` | Collects rollouts from the environment | "Get experience data" |
| `src/agents/pg_agent.py` | Implements policy gradient logic | "Turn rewards into learning signals" |
| `src/networks/policies.py` | Defines the actor / policy network | "Decide what action to take" |
| `src/networks/critics.py` | Defines the value baseline network | "Estimate how good a state is" |

If you want to study the code in order, start with `run.py`, then `utils.py`, then `pg_agent.py`, and finally the two network files.

---

## 1) `src/scripts/run.py`

### Goal
This is the **entry point** for training.

### What each part does

#### Imports and path setup
- Adds `src/` to `sys.path` so Python can import modules like `agents.pg_agent` and `infrastructure.utils`.
- Imports Gym, NumPy, PyTorch, and project utilities.

#### `run_training_loop(logger, args)`
This is the main loop.

- **Seeding**
  - Sets NumPy and PyTorch seeds for reproducibility.
- **GPU setup**
  - Calls `ptu.init_gpu(...)` to choose CPU or GPU.
- **Environment setup**
  - Creates the Gym environment.
  - Detects whether the action space is discrete or continuous.
  - Reads observation and action dimensions.
- **Agent setup**
  - Creates `PGAgent`, which internally creates the policy and maybe the critic.
- **Training loop**
  - Repeats for `args.n_iter` iterations.
  - Collects trajectories using `utils.sample_trajectories`.
  - Converts the list of trajectory dictionaries into lists of arrays.
  - Calls `agent.update(...)` to train the policy and critic.
- **Logging**
  - Periodically collects evaluation rollouts.
  - Logs metrics such as average return and episode length.
  - Optionally saves videos.

### Why it matters in practice
This file is the “manager” of the whole experiment. It does not implement the learning algorithm itself; it only coordinates data collection, training, and logging.

---

## 2) `src/infrastructure/utils.py`

### Goal
This file handles **environment interaction** and **data collection**.

### What each part does

#### `sample_trajectory(env, policy, max_length, render=False)`
Runs **one rollout** in the environment.

- Resets the environment.
- Repeatedly:
  - asks the policy for an action with `policy.get_action(ob)`
  - applies that action with `env.step(ac)`
  - stores observation, action, reward, next observation, and terminal flag
- Stops when:
  - the episode ends naturally, or
  - `max_length` is reached

It returns a dictionary containing arrays for:
- `observation`
- `action`
- `reward`
- `next_observation`
- `terminal`
- `image_obs` if rendering is enabled

#### `sample_trajectories(...)`
Collects **multiple rollouts** until the total number of timesteps is at least a minimum batch size.

- This is how training data batches are built.
- Returns:
  - a list of trajectory dictionaries
  - the total number of steps collected

#### `sample_n_trajectories(...)`
Collects an exact number of trajectories.
- Usually used for evaluation or video saving.

#### `compute_metrics(trajs, eval_trajs)`
Computes logging statistics like:
- average return
- standard deviation of return
- max/min return
- average episode length

#### `convert_listofrollouts(trajs)`
Takes a list of trajectory dictionaries and concatenates them into flat arrays.
- Useful when the algorithm wants all samples in one batch.

### Why it matters in practice
This file is the bridge between the agent and the environment. It turns the policy’s decisions into training data.

---

## 3) `src/agents/pg_agent.py`

### Goal
This file contains the **policy gradient algorithm logic**.

### What each part does

#### `__init__(...)`
Creates the learning components:
- `self.actor`: the policy network
- `self.critic`: optional value network baseline
- stores hyperparameters like:
  - discount factor `gamma`
  - whether to use reward-to-go
  - whether to use GAE
  - whether to normalize advantages

#### `update(obs, actions, rewards, terminals)`
This is the main training step.

##### Step 1: Compute Q-values
- Calls `_calculate_q_vals(rewards)`.
- This turns reward sequences into discounted returns.

##### Step 2: Flatten the batch
- The input is a list of trajectories.
- The code concatenates them into flat arrays so the rest of the pipeline can use vectorized operations.
- After flattening:
  - `obs` becomes a batch of observations
  - `actions` becomes a batch of actions
  - `rewards` becomes a batch of rewards
  - `terminals` becomes a batch of terminal flags
  - `q_values` becomes a batch of Monte Carlo targets

##### Step 3: Estimate advantages
- Calls `_estimate_advantage(...)`.
- This is where you compute what the policy should be rewarded for.

Typical cases:
- **No baseline**: `advantages = q_values`
- **With baseline**: `advantages = q_values - values`
- **With GAE**: use the recursive temporal-difference style advantage estimate

##### Step 4: Update the actor
- Calls `self.actor.update(obs, actions, advantages)`.
- This is where policy parameters are changed.

##### Step 5: Update the critic
- If a critic exists, it is trained to predict the Q-values.
- This reduces variance in future advantage estimates.

#### `_discounted_return(rewards)`
Computes the total discounted return of a whole trajectory.
- Same return value for every timestep in the trajectory.

#### `_discounted_reward_to_go(rewards)`
Computes the return from each timestep onward.
- This is usually lower variance and more informative.

#### `_calculate_q_vals(rewards)`
Chooses between:
- full trajectory return
- reward-to-go

#### `_estimate_advantage(...)`
Produces advantages from Q-values and possibly critic values.
- If no critic: advantage is just Q-value.
- If critic exists: subtract the value estimate.
- If GAE is enabled: use the recursive GAE formula.
- Optionally normalize advantages.

### Why it matters in practice
This file is where the RL math becomes code. It transforms collected rollouts into training signals for the networks.

---

## 4) `src/networks/policies.py`

### Goal
This file defines the **actor / policy network**.

### What each part does

#### `MLPPolicy.__init__(...)`
Builds the policy network depending on the action space type.

- **Discrete action space**
  - builds `self.logits_net`
  - output is a vector of action logits
- **Continuous action space**
  - builds `self.mean_net`
  - creates `self.logstd` as a learnable parameter
  - the policy becomes a Normal distribution

The optimizer is also created here.

#### `get_action(obs)`
Used during environment interaction.
- Converts a single NumPy observation to a torch tensor.
- Runs the policy.
- Samples an action from the resulting distribution.
- Returns it as a NumPy array.

This is the method called inside `sample_trajectory`.

#### `forward(obs)`
Returns a **distribution over actions**.

- For discrete actions:
  - `Categorical(logits=...)`
- For continuous actions:
  - `Normal(mean, std)`

This is a very important design choice: the policy does not directly output an action, it outputs a distribution.

#### `MLPPolicyPG.update(...)`
Performs the policy gradient update.

Typical logic:
- build the action distribution
- compute `log_prob(actions)`
- multiply by `advantages`
- take the negative mean as the loss
- run backpropagation and optimizer step

### Why it matters in practice
This file is the part that actually decides how the agent acts and how the actor is updated.

---

## 5) `src/networks/critics.py`

### Goal
This file defines the **value function / baseline network**.

### What each part does

#### `ValueCritic.__init__(...)`
Builds a neural network that maps observations to a single scalar value `V(s)`.

#### `forward(obs)`
Should return the critic’s prediction for the value of each observation.

#### `update(obs, q_values)`
Trains the critic to predict the Monte Carlo Q-values.

Typical logic:
- predict `V(s)`
- compute MSE loss between prediction and target `q_values`
- backpropagate
- optimizer step

### Why it matters in practice
The critic does not choose actions. Its job is to make the actor’s learning easier by providing a baseline.

---

## How the Files Work Together

### Flow of one training iteration

1. `run.py` starts an iteration.
2. `utils.sample_trajectories(...)` collects rollouts using `policy.get_action(...)`.
3. The rollouts are passed to `PGAgent.update(...)`.
4. `PGAgent` computes Q-values and advantages.
5. `MLPPolicyPG.update(...)` updates the policy.
6. If enabled, `ValueCritic.update(...)` updates the baseline.
7. `run.py` logs performance and continues.

---

## Practical Mental Model

A simple way to think about the code is:

- **`run.py`** = the experiment manager
- **`utils.py`** = data collector
- **`pg_agent.py`** = the RL algorithm
- **`policies.py`** = the actor
- **`critics.py`** = the baseline

---

## Common Confusions

### Why do we need trajectories as lists first?
Because each episode can have a different length. Lists are convenient for collecting them. Later, we concatenate them into flat batches for training.

### Why do we flatten everything?
Neural network training is much easier and faster when the data is in one big batch array.

### Why does the policy output a distribution instead of an action?
Because policy gradient needs `log π(a|s)` for the actions actually taken.
A distribution makes sampling and computing log-probabilities easy.

### Why use a critic?
Because raw Monte Carlo returns are noisy. The critic gives a baseline that reduces variance.

### What is the role of `terminal`?
It marks the end of a trajectory so advantage calculations do not “leak” across episode boundaries.

---

## Short Summary

This project is a complete implementation of policy gradient training:
- `run.py` launches training
- `utils.py` gathers rollouts
- `pg_agent.py` computes targets and advantages
- `policies.py` defines the actor and policy update
- `critics.py` defines the value baseline

If you understand how data flows from the environment into these files and back into the networks, the implementation becomes much easier to reason about.

