# `src/agents/pg_agent.py`

## What this file does
This file contains the **policy gradient algorithm logic**. It is the part that turns collected experience into learning updates for the actor and critic.

## Main parts
- **`__init__(...)`**:
  - creates the policy network (`self.actor`)
  - optionally creates the value network (`self.critic`)
  - stores learning settings like discount factor, reward-to-go, GAE, and advantage normalization
- **`update(...)`**:
  - receives lists of trajectories
  - computes Q-values from rewards
  - flattens all trajectories into one batch
  - estimates advantages
  - updates the policy
  - updates the critic if one exists
- **`_discounted_return(...)`**:
  - computes the total discounted return for a whole trajectory
- **`_discounted_reward_to_go(...)`**:
  - computes return-from-now-to-the-end for each timestep
- **`_calculate_q_vals(...)`**:
  - chooses between full-trajectory return and reward-to-go
- **`_estimate_advantage(...)`**:
  - computes advantages using either no baseline, a value baseline, or GAE

## In plain words
This file is the **brain of the RL algorithm**. It takes rewards from the environment and turns them into the signal used to improve the policy.

