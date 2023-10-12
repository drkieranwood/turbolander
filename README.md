# turbolander
Trials of AI drone landing


ideas from 
https://github.com/marek-robak/Drone-2d-custom-gym-env-for-reinforcement-learning
https://github.com/Code-Bullet/Car-QLearning/tree/master

Agent 1 (1.8 million steps, default SB3 PPO hyperparameters):
Flies a parabolic trajectory to the right. Later became clear this was due to discontinuity between 0 radians and 2pi radians when turning left as level was defined as 0.

Agent 2 (5 million steps, default SB3 PPO hyperparameters):
Flies a sinusoidal trajectory to the right. Same problem as agent 1.

Agent 3 (2 million steps, default SB3 PPO hyperparameters):
Same problem as agent 1. Goal was to test multi-processing, saw notable speed improvement, particularly on RTX 3070.

Agent 4 (5 million steps, default SB3 PPO hyperparameters):
Flies reasonably stably to the goal, struggles with goals near the edge of the environment leading to overshoot and a loop back into the environment to prevent failure. Discontinuity moved to the completely inverted case, attitude now measured from -pi to pi radians.

Pictures and videos to come.

To evaluate a model with rl_zoo3 run the following from a terminal in the directory above turbolander:

```python3 -m rl_zoo3.enjoy --algo ppo --env turbolander-2d-custom-v1 -f turbolander/models/ --exp-id 14 --load-best```

To train a model with rl_zoo3:
```python3 -m rl_zoo3.train --algo ppo --gym-packages turbolander --env turbolander-2d-custom-v1 --progress --conf-file turbolander/hyperparameters_v1.yml -n 2000000 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1
```
