from typing import Literal, TypeAlias

PromptType: TypeAlias = str | list[str] | tuple[str]
TokenReductionType: TypeAlias = Literal['mean', 'last']