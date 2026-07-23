# Adapted from: https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py#L13
from typing import List

from pydantic import BaseModel


class ChatTemplate(BaseModel):
    """
    This is a dataclass for the chat template.

    Args:
        assistant_header(str): The header for the assistant.
        user_header(str): The header for the user.
        system_prompt(str): The system prompt.
        end_of_turn_token(str): The end token of a turn of conversation.
    """

    assistant_header: str | None
    user_header: str | None
    system_prompt: str | None
    end_of_turn_token: str | None
    parser_type: str = "general"
    assistant_pattern_type: str = "general"
    enable_thinking: bool = False


class TemplateRegistry:
    """
    This is a registry for the chat template. Sgl-spec will register some common chat templates here.
    If you have a custom chat template, you can register it via the example below.

    Example:
    ```python
        from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate
        TEMPLATE_REGISTRY.register(
            name="custom",
            template=ChatTemplate(
                assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
                user_header="<|start_header_id|>user<|end_header_id|>",
                system_prompt="You are a helpful assistant.",
                end_of_turn_token="<|eot_id|>"
            )
        )
    ```
    """

    def __init__(self):
        self.templates = {}

    def register(self, name: str, template: ChatTemplate, override: bool = False):
        """
        Register a chat template for a model type.

        Args:
            name(str): The name of the chat template.
            template(ChatTemplate): The chat template.
            override(bool): Whether to override the existing template, default to False
        """
        assert (
            not override and name not in self.templates
        ), f"Chat template for the model type {name} has already been registered"
        self.templates[name] = template

    def get(self, name: str) -> ChatTemplate:
        """
        Get the chat template for a model type.

        Args:
            name(str): The name of the chat template.

        Returns:
            ChatTemplate: The chat template.
        """
        return self.templates[name]

    def get_all_template_names(self) -> List[str]:
        """
        Get all the template names.

        Returns:
            List[str]: The list of template names.
        """
        return list(self.templates.keys())


# global registry
TEMPLATE_REGISTRY = TemplateRegistry()

# Register the common template here
TEMPLATE_REGISTRY.register(
    name="llama3",
    template=ChatTemplate(
        assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
        user_header="<|start_header_id|>user<|end_header_id|>",
        system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        end_of_turn_token="<|eot_id|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="llama4",
    template=ChatTemplate(
        assistant_header="<|header_start|>assistant<|header_end|>\n\n",
        user_header="<|header_start|>user<|header_end|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|eot|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen2-vl",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

# NVIDIA Nemotron-3-Nano (nemotron_h, hybrid Mamba-2/MoE). ChatML-style with a <think></think>
# block auto-injected after the assistant header (model forces a reasoning block); baking it into
# assistant_header keeps SpecForge's rendered training sequence faithful to the model's own template
# so target hidden states are computed on the correct token stream. Default system content is empty.
TEMPLATE_REGISTRY.register(
    name="nemotron-h",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n<think></think>",
        user_header="<|im_start|>user\n",
        system_prompt="",
        end_of_turn_token="<|im_end|>\n",
    ),
)

# NVIDIA Nemotron-3.5-Nano, thinking-ON variant. Same model/tokenizer as `nemotron-h`,
# but for on-policy corpora whose assistant `content` is the target's natural generation
# `<think>REASONING</think>ANSWER`. The tokenizer's own jinja renders the FINAL assistant turn
# with the reasoning intact (and strips reasoning from HISTORY turns) — so rendering is already
# correct under the default GeneralParser. This variant exists ONLY to fix the loss-mask regex:
# set_assistant_pattern (data/parse.py) builds `re.escape(assistant_header) + "([\\s\\S]*?(?:" +
# re.escape(end_of_turn_token) + "|$))"`, so the masked span (group 1) starts right after
# assistant_header. Using the common prefix `<|im_start|>assistant\n<think>` as the header makes
# group 1 cover `REASONING</think>ANSWER<|im_end|>\n` — the non-greedy `[\s\S]*?` is already robust
# to arbitrary reasoning between <think> and </think>, so no custom assistant_pattern_type is needed.
# (The `nemotron-h` header `<think></think>` would fail to match a non-empty reasoning block.)
TEMPLATE_REGISTRY.register(
    name="nemotron-h-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n<think>",
        user_header="<|im_start|>user\n",
        system_prompt="",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi3",
    template=ChatTemplate(
        assistant_header="<|assistant|>\n",
        user_header="<|user|>\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi4",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant<|im_sep|>",
        user_header="<|im_start|>user<|im_sep|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi4-mini",
    template=ChatTemplate(
        assistant_header="<|assistant|>",
        user_header="<|user|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="gpt-oss-naive",
    template=ChatTemplate(
        assistant_header="<|start|>assistant<|channel|>analysis<|message|>",
        user_header="<|start|>user<|message|>",
        system_prompt=None,
        end_of_turn_token="<|end|>",
    ),
)


TEMPLATE_REGISTRY.register(
    name="gpt-oss",
    template=ChatTemplate(
        assistant_header=None,  # the headers are not applicable to openai-harmony's channel tags
        user_header=None,
        system_prompt=None,
        end_of_turn_token=None,
        parser_type="openai-harmony",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek-r1-distill",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        end_of_turn_token=None,
        system_prompt=None,
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen3-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)


TEMPLATE_REGISTRY.register(
    name="qwen3-instruct",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n<think>\n\n</think>\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen3-next-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n<think>\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)

TEMPLATE_REGISTRY.register(
    name="kimi-k2-thinking",
    template=ChatTemplate(
        assistant_header="<|im_assistant|>assistant<|im_middle|>",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
        parser_type="thinking",
        enable_thinking=True,
    ),
)

TEMPLATE_REGISTRY.register(
    name="kimi-k2-instruct",
    template=ChatTemplate(
        assistant_header="<|im_assistant|>assistant<|im_middle|>",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek-v3",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<｜end▁of▁sentence｜>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="ling-flash-2.0",
    template=ChatTemplate(
        assistant_header="<role>ASSISTANT</role>",
        user_header="<role>HUMAN</role>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|role_end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek-v32",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        system_prompt="",
        end_of_turn_token="<｜end▁of▁sentence｜>",
        parser_type="thinking",
        enable_thinking=True,
    ),
)

TEMPLATE_REGISTRY.register(
    name="longcat",
    template=ChatTemplate(
        assistant_header=" ASSISTANT:",
        user_header=" USER:",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="</longcat_s>",
        assistant_pattern_type="longcat",
    ),
)

TEMPLATE_REGISTRY.register(
    name="longcat_xml",
    template=ChatTemplate(
        assistant_header="<longcat_assistant>",
        user_header="<longcat_user>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="</longcat_s>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="glm4",
    template=ChatTemplate(
        assistant_header="<|assistant|></think>",
        user_header="<|user|>",
        system_prompt="",  # GLM tokenizer handles system prompts natively - don't duplicate
        end_of_turn_token="<|endoftext|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="minimax",
    template=ChatTemplate(
        assistant_header="]~b]ai\n",
        user_header="]~b]user\n",
        system_prompt="You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.",
        end_of_turn_token="[e~[\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="gemma",
    template=ChatTemplate(
        assistant_header="<start_of_turn>model\n",
        user_header="<start_of_turn>user\n",
        system_prompt="",
        end_of_turn_token="<end_of_turn>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="gemma4",
    template=ChatTemplate(
        assistant_header="<|turn>model\n",
        user_header="<|turn>user\n",
        system_prompt="",
        end_of_turn_token="<turn|>\n",
    ),
)

# Mistral-Medium-3.5 (and the broader Mistral3 / ministral3 family) uses
# bracketed user turns: [INST]content[/INST]assistant_response</s>. Note the
# closing [/INST] is the marker right before assistant content begins, so
# assistant_header is set to it (not user_footer — the ChatTemplate dataclass
# doesn't have a user_footer field, but the loss-mask regex in
# preprocessing.py uses `end_of_turn_token + assistant_header` to locate
# assistant spans, which gives us "</s>[/INST]" — close enough since the
# first turn has no preceding </s>, and subsequent turns do).
# system_prompt is left empty: Mistral's bundled chat_template.jinja
# inserts [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] natively.
TEMPLATE_REGISTRY.register(
    name="mistral-medium-3-5",
    template=ChatTemplate(
        assistant_header="[/INST]",
        user_header="[INST]",
        system_prompt="",
        end_of_turn_token="</s>",
    ),
)
