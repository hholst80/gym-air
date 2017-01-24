# Atari Image Ram (AIR) Environment

Based on the standard OpenAI AtariEnv but returns a tuple observation (image,
ram).  The image observation is typically a np.array((210, 160, 3),
dtype=np.uint8) ram is an np.array((128,), dtype=np.uint8).

# Install

Clone this repo and install with `pip install -e .` from within the cloned directory.

The environments can be used like:

```
In [1]: import gym

In [2]: import gym_air

In [3]: env = gym.make('MontezumaRevengeDeterministic-image-ram-v0')
[2017-01-24 07:25:18,428] Making new env: MontezumaRevengeDeterministic-image-ram-v0

In [4]: env.observation_space
Out[4]: Tuple(Box(210, 160, 3), Box(128,))

In [5]: env.action_space
Out[5]: Discrete(18)

In [6]: 
```
