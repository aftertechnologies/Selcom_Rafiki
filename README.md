# Selcom Rafiki Assistant

Selcom Rafiki is an interactive AI-powered assistant for Selcom Pesa services. It helps users create accounts, check balances, and answer questions about Selcom Pesa in both English and Swahili. The assistant uses a knowledge base stored in ChromaDB and leverages LlamaIndex and OpenAI for natural language understanding.

## Features

- **Account Creation**: Guides users through creating a Selcom Pesa account.
- **Balance Checking**: Assists users in checking their account balance.
- **Knowledge Base Q&A**: Answers general questions about Selcom Pesa using a vector-based knowledge base.
- **Language Support**: Automatically detects and responds in English or Swahili.
- **Conversation Memory**: Maintains conversation history for context-aware responses.

## Project Structure

```
.env
.gitignore
main.py
selcom_knowledge_base.txt
chroma_db/
    chroma.sqlite3
    <vector store files>
```

- `main.py`: Main application logic for the assistant.
- `selcom_knowledge_base.txt`: Text file containing Selcom Pesa knowledge base information.
- `chroma_db/`: Directory for ChromaDB vector store files.

## Requirements

- Python 3.8+
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [chromadb](https://github.com/chroma-core/chroma)
- [pydantic](https://github.com/pydantic/pydantic)
- [pydantic_ai](https://github.com/pydantic-ai/pydantic-ai)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- OpenAI API key

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:
    ```sh
    pip install llama-index chromadb pydantic pydantic_ai python-dotenv
    ```

3. **Set up environment variables**:
    - Create a `.env` file in the project root.
    - Add your OpenAI API key:
      ```
      OPENAI_API_KEY=your_openai_api_key
      ```

4. **Prepare the knowledge base**:
    - Ensure `selcom_knowledge_base.txt` contains relevant information about Selcom Pesa.

5. **Run the assistant**:
    ```sh
    python main.py
    ```

## Usage

- The assistant will greet you and offer help with Selcom Pesa services.
- Type your questions or requests (e.g., "create account", "check balance").
- You can interact in English or Swahili.
- Type `exit` to quit the assistant.

## Customization

- To update the knowledge base, edit `selcom_knowledge_base.txt` and restart the assistant.
- Modify language responses or add new features in `main.py`.

## License

This project is for