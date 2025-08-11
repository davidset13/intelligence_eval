from pydantic import BaseModel, field_validator
from typing import Optional, Any


class IntelligenceEvalInput(BaseModel):
    agent_name: Optional[Any] = "Not Provided"
    agent_url: str
    prompt_param_name: Optional[Any] = "prompt"
    image_param_name: Optional[Any] = "image"
    agent_params: dict[Any, Any]
    hle: bool = False
    hle_categories: list[str] = ["all"]
    mmlu_pro: bool = False
    mmlu_pro_categories: list[str] = ["all"]
    gpqa: bool = False
    images_enabled: bool = True
    @field_validator("hle_categories")
    @classmethod
    def check_hle_categories(cls, v: list[str]) -> list[str]:
        if "all" in v:
            return ["all"]
        elif len(v) == 0:
            raise ValueError("hle_categories must be a non-empty list")
        else:
            for category in v:
                if category not in ("math", "biology", "computer_science", "other", "physics", "humanities", "chemistry", "engineering"):
                    raise ValueError(f"Invalid category: {category}")
            return [cat[:3].upper() for cat in v]
    @field_validator("mmlu_pro_categories")
    @classmethod
    def check_mmlu_pro_categories(cls, v: list[str]) -> list[str]:
        if "all" in v:
            return ["all"]
        elif len(v) == 0:
            raise ValueError("mmlu_pro_categories must be a non-empty list")
        else:
            for category in v:
                if category not in ("math", "physics", "chemistry", "law", "engineering", "other", "economics", "health", "psychology", "business", "biology", "philosophy", "computer science", "history"):
                    raise ValueError(f"Invalid category: {category}")
            return [cat[:3].upper() for cat in v]




class IntelligenceEvalOutput(BaseModel):
    agent_name: Optional[Any] = "Not Provided"
    hle_accuracy: float | None = None
    hle_ci: tuple[float, float] | None = None
    hle_categories: dict[str, tuple[float, tuple[float, float]]] | None = None
    mmlu_pro_accuracy: float | None = None
    mmlu_pro_ci: tuple[float, float] | None = None
    mmlu_pro_categories: dict[str, tuple[float, tuple[float, float]]] | None = None
    gpqa_accuracy: float | None = None
    gpqa_ci: tuple[float, float] | None = None