"""Validated input types used to start a research run."""

from dataclasses import dataclass


MAX_MODEL_NAME_CHARACTERS = 100
MAX_OBJECTIVE_CHARACTERS = 2_000


@dataclass(frozen=True)
class AgentConfig:
    """Limits and model selection for one agent run."""

    model: str = "scripted"
    max_steps: int = 8
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        if type(self.max_steps) is not int:
            raise TypeError("max_steps must be an integer")
        if type(self.timeout_seconds) is not int:
            raise TypeError("timeout_seconds must be an integer")

        cleaned_model = self.model.strip()
        if not cleaned_model:
            raise ValueError("model must not be empty")
        try:
            cleaned_model.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("model contains invalid Unicode") from error
        if len(cleaned_model) > MAX_MODEL_NAME_CHARACTERS:
            raise ValueError(
                "model must be at most "
                f"{MAX_MODEL_NAME_CHARACTERS} characters"
            )
        if not 1 <= self.max_steps <= 100:
            raise ValueError("max_steps must be between 1 and 100")
        if not 1 <= self.timeout_seconds <= 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600")

        object.__setattr__(self, "model", cleaned_model)


@dataclass(frozen=True)
class ResearchRequest:
    """A research objective paired with its execution configuration."""

    objective: str
    config: AgentConfig

    def __post_init__(self) -> None:
        if not isinstance(self.objective, str):
            raise TypeError("objective must be a string")
        if not isinstance(self.config, AgentConfig):
            raise TypeError("config must be an AgentConfig")

        cleaned_objective = self.objective.strip()
        if not cleaned_objective:
            raise ValueError("objective must not be empty")
        try:
            cleaned_objective.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("objective contains invalid Unicode") from error
        if len(cleaned_objective) > MAX_OBJECTIVE_CHARACTERS:
            raise ValueError(
                "objective must be at most "
                f"{MAX_OBJECTIVE_CHARACTERS} characters"
            )

        object.__setattr__(self, "objective", cleaned_objective)
