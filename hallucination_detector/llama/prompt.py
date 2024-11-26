class LlamaPrompt:
  """
   Class to represent a prompt for the Llama model, which is made of a system prompt,
   which sets the general context according to which the AI should respond,
   and a user prompt, which is the text that the AI should respond to.
  """
  user_prompt: str
  system_prompt: str

  def __init__(self, user_prompt, system_prompt="You are a helpful AI assistant."):
    self.user_prompt = user_prompt
    self.system_prompt = system_prompt

  def __str__(self) -> str:
      # From: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#-instruct-model-prompt-
      return ''.join([
          "<|begin_of_text|>",
          f"<|start_header_id|>system<|end_header_id|>{self.system_prompt}<|eot_id|>",
          f"<|start_header_id|>user<|end_header_id|>{self.user_prompt}<|eot_id|>",
          "<|start_header_id|>assistant<|end_header_id|>"
      ])