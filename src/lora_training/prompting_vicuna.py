"""
Vendored from openvla: Vicuna-v1.5 style chat prompt for OpenVLA fine-tuning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_prompt(self) -> str: ...


SYS_PROMPTS = {
    "openvla": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
}


class VicunaV15ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = (
            SYS_PROMPTS.get(self.model_family, SYS_PROMPTS["openvla"])
            if system_prompt is None
            else system_prompt
        ).strip() + " "

        self.bos, self.eos = "<s>", "</s>"
        self.wrap_human = lambda msg: f"USER: {msg} ASSISTANT: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        # Keep "<image>" — OpenVLA / Prismatic use it to align pixel_features in the sequence.
        message = message.strip()

        if self.turn_count == 0:
            wrapped_message = self.system_prompt + self.wrap_human(message)
        elif (self.turn_count % 2) == 0:
            wrapped_message = self.wrap_human(message)
        else:
            wrapped_message = self.wrap_gpt(message)

        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_prompt(self) -> str:
        return self.prompt.removeprefix(self.bos).rstrip()
