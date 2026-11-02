"""Environment factory for LunarLander-v3 with configurable reward wrappers.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym

from rocketmind.reward_wrappers import (
    CorrectedPotentialRewardWrapper,
    HistoricalPotentialRewardWrapper,
    HoverRewardWrapper,
)


def make_env(
    env_id: str = "LunarLander-v3",
    wrapper_type: str = "none",
    render_mode: Optional[str] = None,
    **wrapper_kwargs: Any,
) -> gym.Env:
    """Instantiate a Gymnasium environment with optional reward wrappers.
    
    Args:
        env_id: Gymnasium environment ID (default: LunarLander-v3).
        wrapper_type: One of 'none', 'hover' (deliberate hack),
                      'potential_historical' (Exp C implementation),
                      or 'potential_corrected' (rigorous PBRS).
        render_mode: None, 'human', or 'rgb_array'.
        **wrapper_kwargs: Extra keyword arguments forwarded to the wrapper.
        
    Returns:
        gym.Env instance.
    """
    env = gym.make(env_id, render_mode=render_mode)

    wrapper_type = wrapper_type.lower()
    if wrapper_type in ("none", "baseline"):
        return env
    elif wrapper_type in ("hover", "reward_hack", "hover_bonus"):
        bonus = wrapper_kwargs.get("bonus_amount", 1.0)
        return HoverRewardWrapper(env, bonus_amount=bonus)
    elif wrapper_type in ("potential_historical", "progress"):
        gamma = wrapper_kwargs.get("gamma", 0.99)
        scale = wrapper_kwargs.get("shaping_scale", 10.0)
        return HistoricalPotentialRewardWrapper(env, gamma=gamma, shaping_scale=scale)
    elif wrapper_type in ("potential_corrected", "pbrs"):
        gamma = wrapper_kwargs.get("gamma", 0.99)
        scale = wrapper_kwargs.get("shaping_scale", 5.0)
        return CorrectedPotentialRewardWrapper(env, gamma=gamma, shaping_scale=scale)
    else:
        raise ValueError(
            f"Unknown wrapper_type: '{wrapper_type}'. "
            f"Expected one of ['none', 'hover', 'potential_historical', 'potential_corrected']."
        )

