from gym.envs.registration import register

register(
    id="Crosswalk_hybrid_multi_opti7_3-v0",
    entry_point="Environments.Env_hybrid_multi_opti7_3:Crosswalk_hybrid_multi_opti7_3",
    max_episode_steps=100,
    reward_threshold=100.0,
)
