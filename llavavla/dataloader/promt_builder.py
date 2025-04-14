from typing import Optional, Union, Dict, Any, List
from PIL import Image

class QwenVLPromptHelper_v1:
    def __init__( #@Jinhui 考虑这里是否要和模型解偶合
        self,
        processor,
        system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    ):
        """
        Helper for building and processing Qwen-VL style prompts.

        Args:
            processor: Qwen processor (AutoProcessor from Qwen-VL)
            system_prompt: Optional custom system prompt string
        """
        self.processor = processor
        self.system_prompt = system_prompt.strip()

    def build_conversation(
        self,
        instruction: str,
        image: Optional[Image.Image] = None,
        answer: Optional[str] = "",
    ) -> List[Dict[str, Union[str, List[Dict[str, Any]]]]]:
        """
        Build a conversation in Qwen-VL format.

        Args:
            instruction (str): Task instruction
            image (Optional): PIL image object or None placeholder
            answer (str): GPT's answer; empty string for inference

        Returns:
            A list of conversation messages
        """
        conversation = []

        if self.system_prompt:
            conversation.append({
                "role": "system",
                "content": self.system_prompt
            })

        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"What action should the robot take to {instruction.strip().lower()}?"},
                {"type": "image", "image": image}  # ← usually None before actual processing
            ]
        })

        conversation.append({
            "role": "gpt",
            "content": answer if answer else ""
        })

        return conversation

    def build_multimodal_inputs(
        self,
        conversation: List[Dict[str, Any]],
        image: Image.Image,
        add_generation_prompt: bool = True,
        return_prompt_text: bool = False,
        ) -> Union[Dict[str, Any], tuple]:
        """
        Apply Qwen chat template and tokenize with image.

        Args:
            multimodal_input (List[Dict]): The structured conversation (text + image placeholder).
            image (PIL.Image): Image input (resized inside).
            add_generation_prompt (bool): Whether to append <|im_start|>assistant for generation.
            return_prompt_text (bool): Whether to return raw string prompt.

        Returns:
            inputs (Dict[str, Tensor]) or (inputs, prompt_text)
        """
        prompt_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

        image = image.resize((224, 224))  # Resize to Qwen-VL default input
        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        return (inputs, prompt_text) if return_prompt_text else inputs

from typing import Optional, Union, Dict, Any, List
from PIL import Image


class QwenVLPromptHelper:
    def __init__(
        self,
        processor,
        system_prompt: str = "You are a helpful assistant."
    ):
        """
        用于构建 Qwen-VL 定位类任务输入（坐标点击/关键点定位等）。

        Args:
            processor: Qwen 的 AutoProcessor 实例
            system_prompt: System 指令，可指定为用于 JSON 格式输出等任务
        """
        self.processor = processor
        self.system_prompt = system_prompt.strip()
        self.cognition_emoj = "🔍"
        self.cognition_token_ids = self.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]
    def build_conversation(
        self,
        instruction: str,
        image: Optional[Image.Image] = None,
        answer: Optional[str] = "",
        output_format: str = "coord"
    ) -> List[Dict[str, Union[str, List[Dict[str, Any]]]]]:
        """
        构建符合 Qwen-VL 多模态格式的对话内容。

        Args:
            instruction: 文本指令，如 “请指出红色按钮的位置”
            image: 输入图像对象（可以为 None 占位）
            answer: 若用于训练则提供真实答案（如 "点击点为：[120, 256]"）
            output_format: 输出格式（"coord"=纯坐标, "json"=结构化, "raw"=自由文本）

        Returns:
            conversation (List): 多模态对话格式，用于 apply_chat_template
        """
        conversation = []

        if self.system_prompt:
            if output_format == "json":
                system = self.system_prompt + " Respond with only JSON format: {\"point\": [x, y]}"
            else:
                system = self.system_prompt
            conversation.append({"role": "system", "content": system})

        # 用户指令
        user_msg = [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{instruction.strip().lower()}."},
            {"type": "text", "text": f"{self.cognition_emoj}"}, #TODO add more grounding here
            
        ]
        conversation.append({"role": "user", "content": user_msg})

        # GPT回复（用于训练时提供正确答案；推理时为空）
        conversation.append({"role": "gpt", "content": answer if answer else ""})

        return conversation

    def build_multimodal_inputs(
        self,
        conversation: List[Dict[str, Any]],
        image: Image.Image,
        add_generation_prompt: bool = True,
        return_prompt_text: bool = False,
    ) -> Union[Dict[str, Any], tuple]:
        """
        将对话结构转换为模型可用的输入（用于 tokenizer + image）。

        Args:
            conversation: 对话内容，由 build_conversation 构造
            image: PIL image 输入
            add_generation_prompt: 是否添加 assistant 回答起始标志
            return_prompt_text: 是否返回原始文本 prompt

        Returns:
            model_inputs (Dict)，或 (model_inputs, prompt_text)
        """
        prompt_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

        image = image.resize((224, 224))  # RLDS 默认分辨率
        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        return (inputs, prompt_text) if return_prompt_text else inputs

    def encode_inputs(
        self,
        instruction: str,
        image: Image.Image,
        answer: Optional[str] = "",
        output_format: str = "coord",
        return_prompt_text: bool = False,
        add_generation_prompt: bool = True
    ) -> Union[Dict[str, Any], tuple]:
        """
        高级封装，一步构造 prompt + tokenizer 输出。

        Args:
            instruction: 任务指令，如 “抓住那个红色的瓶子”
            image: PIL image 输入
            answer: 若为训练则提供 ground truth
            output_format: “coord” / “json” / “raw”
            return_prompt_text: 是否返回原始文本 prompt
            add_generation_prompt: 是否添加生成提示符

        Returns:
            Dict 或 (Dict, str): inputs 以及 prompt_text（可选）
        """
        conv = self.build_conversation(
            instruction=instruction,
            image=None,  # 对话中的占位，真实 image 在 tokenizer 输入中提供
            answer=answer,
            output_format=output_format
        )
        return self.build_multimodal_inputs(
            conversation=conv,
            image=image,
            add_generation_prompt=add_generation_prompt,
            return_prompt_text=return_prompt_text
        )

if __name__ == "__main__":
    from PIL import Image
    from modelscope import AutoProcessor

    # 加载 Qwen-VL 处理器
    processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat")

    helper = QwenVLPromptHelper(processor)

    img = Image.open("example.jpg")
    instruction = "press the red button"

    inputs, prompt = helper.encode_inputs(
        instruction=instruction,
        image=img,
        output_format="json",  # 或 "coord"
        return_prompt_text=True
    )

    print(prompt)
    # -> 会输出完整 prompt，包括 system/user/assistant 结构
    # -> inputs 可直接传入 model.generate()
