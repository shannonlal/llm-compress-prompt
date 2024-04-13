from typing import List
from .token import get_token_length
from .control_context_budget import control_context_budget
from .control_sentence_budget import control_sentence_budget

class LLMPromptCompressor:
    """
    LLMPromptCompressor is designed for compressing prompts based on a given language model.

    This class is very similar to the original PromptCompressor class, but it is designed to work with the LLMLingua model;
    however, it does not require a LLM model and instead it relies on 3rd LLM (OpenaI, Claude, etc) to support the compression.
    
    
    The implementation is based Microsoft's https://github.com/microsoft/LLMLingua which was defined in the 
    paper "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models". Jiang, Huiqiang, Qianhui Wu,
    Chin-Yew Lin, Yuqing Yang, and Lili Qiu. "Llmlingua: Compressing prompts for accelerated inference of large language models."
    arXiv preprint arXiv:2310.05736 (2023).

    Args:
        rank_method (str): The rank_method used to support the compression. Default is "openai".
        concurrent_requests (int): The number of concurrent requests to be made to the LLM API. Default is 1.
        llm_api_config (dict, optional): A dictionary containing configuration for for the different LLMs APIs that may be used in conjunction with the model. Default is an empty dictionary.

    Example:
        >>> compress_method = LLMPromptCompressor(rank_method="openai", concurrent_requests=1, llm_api_config={})
        >>> context = ["This is the first context sentence.", "Here is another context sentence."]
        >>> result = compress_method.compress_prompt(context)
        >>> print(result["compressed_prompt"])
        # This will print the compressed version of the context.

    """

    def __init__(
        self,
        rank_method: str,
        concurrent_requests: int = 1,
        llm_api_config: dict = {},
    ):
        self.rank_method = rank_method
        self.concurrent_requests = concurrent_requests
        self.llm_api_config = llm_api_config


    def compress_prompt(
        self,
        context: List[str],
        instruction: str = "",
        question: str = "",
        rate: float = 0.5,
        target_token: float = -1,
        use_sentence_level_filter: bool = False,
        use_context_level_filter: bool = True,
        use_token_level_filter: bool = True,
        concate_question: bool = True,
    ):
        """
        Compresses the given context.

        Args:
            context (List[str]): List of context strings that form the basis of the prompt.
            instruction (str, optional): Additional instruction text to be included in the prompt. Default is an empty string.
            question (str, optional): A specific question that the prompt is addressing. Default is an empty string.
            rate (float, optional): The maximum compression rate target to be achieved. The compression rate is defined
                the same as in paper "Language Modeling Is Compression". Delétang, Grégoire, Anian Ruoss, Paul-Ambroise Duquenne,
                Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya et al. "Language modeling is compression."
                arXiv preprint arXiv:2309.10668 (2023):
                .. math::\text{Compression Rate} = \frac{\text{Compressed Size}}{\text{Raw Size}}
                Default is 0.5. The actual compression rate is generally lower than the specified target, but there can be
                fluctuations due to differences in tokenizers. If specified, it should be a float less than or equal
                to 1.0, representing the target compression rate.
            target_token (float, optional): The maximum number of tokens to be achieved. Default is -1, indicating no specific target.
                The actual number of tokens after compression should generally be less than the specified target_token, but there can
                be fluctuations due to differences in tokenizers. If specified, compression will be based on the target_token as
                the sole criterion, overriding the ``rate``.
            use_sentence_level_filter (bool, optional): Whether to apply sentence-level filtering in compression. Default is False.
            use_context_level_filter (bool, optional): Whether to apply context-level filtering in compression. Default is True.
            concate_question (bool, optional): Whether to concatenate the question to the compressed prompt. Default is True.

        Returns:
            dict: A dictionary containing:
                - "compressed_prompt" (str): The resulting compressed prompt.
                - "origin_tokens" (int): The original number of tokens in the input.
                - "compressed_tokens" (int): The number of tokens in the compressed output.
                - "ratio" (str): The compression ratio achieved, calculated as the original token number divided by the token number after compression.
                - "rate" (str): The compression rate achieved, in a human-readable format.
                - "saving" (str): Estimated savings in GPT-4 token usage.
        """
        assert (
            rate <= 1.0
        ), "Error: 'rate' must not exceed 1.0. The value of 'rate' indicates compression rate and must be within the range [0, 1]."


        origin_tokens = get_token_length("\n\n".join([instruction] + context + [question]).strip())

        context_tokens_length = [get_token_length(c) for c in context]
        instruction_tokens_length, question_tokens_length = get_token_length(
            instruction
        ), get_token_length(question)
        if target_token == -1:
            target_token = (
                (
                    instruction_tokens_length
                    + question_tokens_length
                    + sum(context_tokens_length)
                )
                * rate
                - instruction_tokens_length
                - (question_tokens_length if concate_question else 0)
            )

        if len(context) > 1 and use_context_level_filter:
            context, dynamic_ratio, context_used = control_context_budget(
                context,
            )


        if use_sentence_level_filter:
            context, segments_info = control_sentence_budget(
                context,
                target_token,
            )

        compressed_prompt = "\n\n".join(context)

        res = []
        if instruction:
            res.append(instruction)
        if compressed_prompt.strip():
            res.append(compressed_prompt)
        if question and concate_question:
            res.append(question)

        compressed_prompt = "\n\n".join(res)

        compressed_tokens = get_token_length(compressed_prompt)
        saving = (origin_tokens - compressed_tokens) * 0.06 / 1000
        ratio = 1 if compressed_tokens == 0 else origin_tokens / compressed_tokens
        rate = 1 / ratio
        return {
            "compressed_prompt": compressed_prompt,
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": f"{ratio:.1f}x",
            "rate": f"{rate * 100:.1f}%",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
        }