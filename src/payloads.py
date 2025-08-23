from pydantic import BaseModel, field_validator
from typing import Optional, Any
import pandas as pd
import os
from dotenv import load_dotenv


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
    livebench: bool = False
    livebench_categories: list[str] = ["all"]
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
    @field_validator("livebench_categories")
    @classmethod
    def check_livebench_categories(cls, v: list[str]) -> list[str]:
        if "all" in v:
            return ["all"]
        elif len(v) == 0:
            raise ValueError("mmlu_pro_categories must be a non-empty list")
        else:
            for category in v:
                if category not in ("reasoning", "data_analysis", "instruction_following", "math", "language"):
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
    livebench_accuracy: float | None = None
    livebench_ci: tuple[float, float] | None = None
    livebench_categories: dict[str, tuple[float, tuple[float, float]]] | None = None


def get_proper_gpt_dataset() -> tuple[pd.DataFrame, str]:
    if not load_dotenv(os.path.join(os.getcwd(), "cmn_pckgs", "puppeteer", ".private.env")):
        raise FileNotFoundError("Private Environment File Not Found")
    
    gpt_model: str = os.getenv("GPT_MODEL") or ""
    if gpt_model == "GPT-5-FAST":
        gpt_fast_ans = pd.read_csv(os.path.join(os.getcwd(), "agent_ans", "chatgpt-gpt5-fast.csv"), encoding="utf-8")
        gpt_ds_name = "chatgpt-gpt5-fast"
        return gpt_fast_ans, gpt_ds_name
    elif gpt_model == "GPT-5-THINK":
        gpt_think_ans = pd.read_csv(os.path.join(os.getcwd(), "agent_ans", "chatgpt-gpt5-think.csv"), encoding="utf-8")
        gpt_ds_name = "chatgpt-gpt5-think"
        return gpt_think_ans, gpt_ds_name
    elif gpt_model == "GPT-4o":
        gpt_4o_ans = pd.read_csv(os.path.join(os.getcwd(), "agent_ans", "chatgpt-gpt4o.csv"), encoding="utf-8")
        gpt_ds_name = "chatgpt-gpt4o"
        return gpt_4o_ans, gpt_ds_name
    else:
        raise ValueError("GPT Model Not Supported")

(ds, ds_name) = get_proper_gpt_dataset()
gpt_dataset = ds
gpt_ds_name = ds_name