from pydantic import BaseModel
from typing import Optional, Any

class IntelligenceEvalInput(BaseModel):
    agent_name: Optional[Any] = "Not Provided"
    agent_url: str
    prompt_param_name: Optional[Any] = "prompt"
    image_param_name: Optional[Any] = "image"
    agent_params: dict[Any, Any]
    hle: bool = False
    mmlu_pro: bool = False
    gpqa: bool = False
    images_enabled: bool = True


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