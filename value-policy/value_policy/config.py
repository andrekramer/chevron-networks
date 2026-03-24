from dataclasses import dataclass


@dataclass(slots=True)
class EnvConfig:
    horizon: int = 8
    p_reversal: float = 0.5
    reversal_min_step: int = 3
    reversal_max_step: int = 5
    cue_prob: float = 0.60
    wait_penalty: float = -0.01
    reversal_cue_delay: int = 1
    distractor_dim: int = 4

    @property
    def obs_dim(self) -> int:
        return 2 + self.distractor_dim

    @property
    def action_dim(self) -> int:
        return 3


@dataclass(slots=True)
class PPOConfig:
    num_envs: int = 128
    updates: int = 200
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 7
    device: str = "cpu"


@dataclass(slots=True)
class ModelConfig:
    model_name: str = "structured_chevron"
    hidden_size: int = 64
    encoder_size: int = 64
    uncertainty_scale: bool = True
    gated_policy: bool = False
