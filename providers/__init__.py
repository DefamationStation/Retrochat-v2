import importlib
import os

class DynamicProviderRegistry:
    def __init__(self, providers_directory):
        self.providers_directory = os.path.abspath(providers_directory)
        self.providers = {}
        self._load_providers()

    def _load_providers(self):
        if not os.path.exists(self.providers_directory):
            raise FileNotFoundError(f"The directory {self.providers_directory} does not exist.")
        for filename in os.listdir(self.providers_directory):
            if filename.endswith(".py") and filename != "__init__.py" and filename != "provider_base.py":
                module_name = filename[:-3]
                module = importlib.import_module(f"providers.{module_name}")
                # Convert filename to class name convention (e.g., "openai_gpt4o.py" -> "OpenaiGpt4oProvider")
                provider_class_name = ''.join([part.capitalize() for part in module_name.split('_')]) + "Provider"
                try:
                    provider_class = getattr(module, provider_class_name)
                    self.providers[provider_class_name] = provider_class  # Use full class name as the key
                except AttributeError:
                    print(f"Warning: {provider_class_name} not found in {module_name}. Skipping module.")

    def get_provider(self, name, *args, **kwargs):
        provider_class = self.providers.get(name)
        if not provider_class:
            raise ValueError(f"Provider '{name}' not found.")
        return provider_class(*args, **kwargs)
