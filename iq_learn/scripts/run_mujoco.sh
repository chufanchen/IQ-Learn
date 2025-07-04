#!/usr/bin/env bash

# Train on Mujoco environments (Default: Use 1 expert demo)
# Set expert.demos=1 for using one expert demo.

# Set working directory to iq_learn
cd ..

# Hopper-v2
python train_iq.py env=hopper agent=sac expert.demos=1 method.loss=v0 method.regularize=True agent.actor_lr=3e-5 seed=0

# HalfCheetah-v2
python train_iq.py env=cheetah agent=sac expert.demos=1 method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0

# Ant-v2
python train_iq.py env=ant agent=sac expert.demos=1 method.loss=value method.regularize=True agent.actor_lr=3e-05 agent.init_temp=0.001 seed=0

# Walker2d-v2
python train_iq.py env=walker agent=sac expert.demos=1 method.loss=v0 method.regularize=True agent.actor_lr=3e-05 seed=0

# Humanoid-v2
python train_iq.py env=humanoid agent=sac expert.demos=1 method.loss=v0 method.regularize=True agent.actor_lr=3e-05 seed=0 agent.init_temp=1

python train_iq.py agent=sac agent.actor_lr=3e-05 agent.critic_lr=0.0003 agent.init_temp=0.001 env=hopper env.learn_steps=1e6 env.demos=10 method.loss=value method.regularize=True num_actor_updates=1 num_seed_steps=0 train.batch=256 train.soft_update=True train.use_target=True seed=2


python test_iq.py env=hopper agent=sac  seed=0 eval.policy=trained_policies/hopper/test2/ q_net._target_=agent.sac_models.DoubleQCritic