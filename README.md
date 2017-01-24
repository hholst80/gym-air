# Atari Image Ram (AIR) Environment

Based on the standard OpenAI AtariEnv but returns a tuple observation (image,
ram).  The image observation is typically a np.array((210, 160, 3),
dtype=np.uint8) ram is an np.array((128,), dtype=np.uint8).
