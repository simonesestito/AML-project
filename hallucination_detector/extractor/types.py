from typing import Literal

type PromptType = str | list[str] | tuple[str]
type TokenReductionType = Literal['mean', 'last']