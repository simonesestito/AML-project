class LlamaPrompt:
  """
   Class to represent a prompt for the Llama model, which is made of a system prompt,
   which sets the general context according to which the AI should respond,
   and a user prompt, which is the text that the AI should respond to.
  """
  user_prompt: str
  system_prompt: str
  with_prefix: bool
  with_suffix: bool


  def __init__(self, user_prompt, system_prompt: str = "You are a helpful AI assistant.", with_prefix: bool = True, with_suffix: bool = True):
    self.user_prompt = user_prompt
    self.system_prompt = system_prompt
    self.with_prefix = with_prefix
    self.with_suffix = with_suffix


  def __str__(self) -> str:
      # From: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#-instruct-model-prompt-
      prefix = ''.join([
          "<|begin_of_text|>",
          f"<|start_header_id|>system<|end_header_id|>{self.system_prompt}<|eot_id|>",
          "<|start_header_id|>user<|end_header_id|>",
      ]) if self.with_prefix else ''

      suffix = ''.join([
          "<|eot_id|>",
          "<|start_header_id|>assistant<|end_header_id|>"
      ]) if self.with_suffix else ''

      return prefix + self.user_prompt + suffix
