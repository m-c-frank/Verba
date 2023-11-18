import requests
from goldenverba.components.component import VerbaComponent

class OllamaGenerator(VerbaComponent):
    """
    Implementation of the Verba Generator for the Ollama model, treating the response as a single complete response.
    """

    def __init__(self):
        super().__init__()
        self.ollama_url = "http://localhost:11434/api/generate"  # Ollama API URL

    def generate(
        self, 
        queries: list[str], 
        context: list[str], 
        conversation: dict = {}
    ) -> str:
        """
        Generate an answer using the Ollama model.
        """
        prompt = self.prepare_prompt(queries, context, conversation)
        response = requests.post(
            self.ollama_url, 
            json={'model': 'llama2', 'prompt': prompt, 'format': 'json'}
        )
        result = response.json()
        return result.get("response", "")

    def prepare_messages(
        self, queries: list[str], context: list[str], conversation: dict[str, str]
    ) -> dict[str, str]:
        """
        Prepares a list of messages formatted for a Retrieval Augmented Generation chatbot system, including system instructions, previous conversation, and a new user query with context.

        @parameter queries: A list of strings representing the user queries to be answered.
        @parameter context: A list of strings representing the context information provided for the queries.
        @parameter conversation: A list of previous conversation messages that include the role and content.

        @returns A list of message dictionaries formatted for the chatbot. This includes an initial system message, the previous conversation messages, and the new user query encapsulated with the provided context.

        Each message in the list is a dictionary with 'role' and 'content' keys, where 'role' is either 'system' or 'user', and 'content' contains the relevant text. This will depend on the LLM used.
        """
        messages = [
            {
                "role": "system",
                "content": f"You are a Retrieval Augmented Generation chatbot. Please answer user queries only their provided context. If the provided documentation does not provide enough information, say so. If the answer requires code examples encapsulate them with ```programming-language-name ```. Don't do pseudo-code.",
            }
        ]

        for message in conversation:
            messages.append({"role": message.type, "content": message.content})

        query = " ".join(queries)
        user_context = " ".join(context)

        messages.append(
            {
                "role": "user",
                "content": f"Please answer this query: '{query}' with this provided context: {user_context}",
            }
        )

        return messages

