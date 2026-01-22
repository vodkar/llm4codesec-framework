from abc import ABC, abstractmethod

from pydantic import BaseModel


class IPromptGenerator(ABC, BaseModel):
    system_prompt_template: str
    user_prompt_template: str
    template_values: dict[str, str]

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Generate the system prompt.

        Returns:
            str: Rendered system prompt.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_prompt(self, template_values: dict[str, str]) -> str:
        """Generate the user prompt.

        Args:
            template_values (dict[str, str]): Template values to override.

        Returns:
            str: Rendered user prompt.
        """
        raise NotImplementedError


class DefaultPromptGenerator(IPromptGenerator):
    def get_system_prompt(self) -> str:
        """Generate the system prompt.

        Returns:
            str: Rendered system prompt.
        """
        return self.system_prompt_template.format(**self.template_values)

    def get_user_prompt(self, template_values: dict[str, str]) -> str:
        """Generate the user prompt.

        Args:
            template_values (dict[str, str]): Template values to override.

        Returns:
            str: Rendered user prompt.
        """
        return self.user_prompt_template.format(
            **self.template_values, **template_values
        )
