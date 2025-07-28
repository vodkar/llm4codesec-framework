from abc import ABC, abstractmethod

from pydantic import BaseModel


class IPromptGenerator(ABC, BaseModel):
    system_prompt_template: str
    user_prompt_template: str
    template_values: dict[str, str]

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Generate system prompt based on the template and values."""
        pass

    @abstractmethod
    def get_user_prompt(self, template_values: dict[str, str]) -> str:
        """Generate user prompt based on the template and values."""
        pass


class DefaultPromptGenerator(IPromptGenerator):
    def get_system_prompt(self):
        """Generate system prompt based on the template and values."""
        return self.system_prompt_template.format(**self.template_values)

    def get_user_prompt(self, template_values: dict[str, str]):
        """Generate user prompt based on the template and values."""
        return self.user_prompt_template.format(
            **self.template_values, **template_values
        )
