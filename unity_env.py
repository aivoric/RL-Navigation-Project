from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(file_name="Banana.app", no_graphics=True, worker_id=2, seed=1, side_channels=[])

