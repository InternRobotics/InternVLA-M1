
from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

SYS_PROMPTS = {
    "prismatic": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "openvla": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}


class QwenPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        self.system_prompt = (SYS_PROMPTS[model_family] if system_prompt is None else self.system_prompt).strip()

        # Note =>> Qwen Tokenizer is an instance of `Qwen2Tokenizer(Fast)`
        #      =>> By default, there is *no* <BOS> token. we add <EOS> manually.
        self.bos = self.start = "<|im_start|>"  # NOTE this is not used
        self.eos = "<|endoftext|>"

        self.end = "<|im_end|>"

        # Get role-specific "wrap" functions
        #   =>> Note that placement of <bos>/<eos> were based on experiments generating from Phi-2 in Input/Output mode
        self.wrap_system = lambda msg: f"{self.start}system\n{msg}{self.end}\n"
        self.wrap_human = lambda msg: f"{self.start}user\n{msg}{self.end}\n{self.start}assistant\n"
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.end}\n"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "first" input --> add an optional system prompt to the beginning.
        if self.turn_count == 0 and self.system_prompt is not None:
            self.prompt += self.wrap_system(self.system_prompt)

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy

    def get_prompt(self) -> str:
        # add EOS if we ended on a "gpt" role (turns is a multiple of 2)
        if self.turn_count % 2 == 0:
            # remove the newline before EOS
            assert self.prompt[-1] == "\n", f"malformed prompt ({self.prompt}) missing newline before EOS append!"
            return self.prompt[:-1] + self.eos

        return self.prompt
