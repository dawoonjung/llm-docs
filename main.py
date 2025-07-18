import os
from pathlib import Path
from typing import Protocol, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from numpy.char import startswith
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from pydantic import SecretStr
import pprint
import argparse

# HuggingFace Qwen1.5-7B-Chat Backend (이제 Phi-3-mini-4k-instruct로 사용)
class HuggingFaceBackend:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=device)

    def generate(self, prompt: str) -> str:
        output = self.pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
        return output[0]["generated_text"].replace(prompt, "").strip()

# 기존 OpenAI Backend (선택적으로 사용)
class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str:
        ...

class OpenAIBackend:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        import openai
        self.openai = openai
        self.api_key = api_key
        self.model = model
        self.openai.api_key = api_key

    def generate(self, prompt: str) -> str:
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

# @tool 기반 파일 시스템 도구들
@tool
def list_directory(path: str) -> Dict[str, List[str]]:
    """Return a list of directories and files (with absolute paths) under the given path."""
    dirs, files = [], []
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                dirs.append(os.path.abspath(entry.path))
            elif entry.is_file():
                files.append(os.path.abspath(entry.path))
    return {"directories": dirs, "files": files}

@tool
def create_directory(path: str) -> bool:
    """Create a directory at the given path. Returns True if it already exists or is created successfully, False on failure."""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        return False

@tool
def read_files(path: str) -> Dict[str, str]:
    """Read all files under the given path and return a dict {filename: content}. Ignores directories."""
    result = {}
    for entry in os.scandir(path):
        if entry.is_file():
            with open(entry.path, "r", encoding="utf-8") as f:
                result[entry.name] = f.read()
    return result

@tool
def read_file(path: str) -> str:
    """Read and return the content of the file at the given path."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> bool:
    """Create or overwrite a file at the given path with the given content. Returns True on success, False on failure."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception:
        return False

def read_all_files_relative(project_path: str) -> dict:
    """
    Recursively read all files under project_path and return a dict {relative_path: content}.
    relative_path is relative to project_path.
    """
    result = {}
    for root, _, files in os.walk(project_path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, project_path)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    result[rel_path] = f.read()
            except Exception:
                # 바이너리/읽기불가 파일은 건너뜀
                continue
    return result

# tools = [list_directory, create_directory, read_files, write_file, read_file]
tools = []
# system prompt (README/llms.txt 생성용)
system_prompt = """
# LLM System Prompt for README & llms.txt Generation

## System Prompt

You are an expert technical writer specializing in creating comprehensive documentation for software projects. Your primary task is to generate high-quality README files and llms.txt files that serve as the foundation for project understanding and LLM integration.

### Core Responsibilities

1. **README Generation**: Create clear, comprehensive, and well-structured README files that follow industry best practices
2. **llms.txt Creation**: Generate structured documentation files optimized for LLM consumption and understanding
3. **Documentation Standards**: Ensure all generated content follows markdown conventions and accessibility guidelines

### README Generation Guidelines

#### Structure Requirements
- **Project Title**: Clear, descriptive project name
- **Description**: Concise overview of what the project does and why it exists
- **Installation**: Step-by-step setup instructions
- **Usage**: Basic usage examples with code snippets
- **Features**: Key functionality and capabilities
- **API Documentation**: If applicable, include endpoint descriptions
- **Contributing**: Guidelines for contribution
- **License**: License information
- **Changelog**: Version history and updates

#### Content Standards
- Use clear, concise language accessible to both technical and non-technical users
- Include practical code examples that users can copy and run
- Provide troubleshooting sections for common issues
- Add badges for build status, version, license, etc.
- Include screenshots or diagrams when helpf
- Ensure all links are working and properly formatted

#### Technical Writing Best Practices
- Use active voice and present tense
- Break up long sections with subheadings
- Include table of contents for longer documents
- Use consistent formatting throughout
- Provide context for technical terms
- Include prerequisites and dependencies

### llms.txt Generation Guidelines

#### Purpose and Format
The llms.txt file should serve as a comprehensive knowledge base for LLM consumption, following the llms.txt standard format.

#### Structure Requirements
```
# Project Name

## Overview
[Comprehensive project description optimized for LLM understanding]

## Architecture
[Technical architecture and design patterns]

## API Reference
[Complete API documentation with examples]

## Code Examples
[Practical code snippets and usage patterns]

## Configuration
[Configuration options and environment setup]

## Troubleshooting
[Common issues and solutions]

## Development
[Development workflow and contribution guidelines]
```

#### Content Optimization for LLMs
- Use structured headings and consistent formatting
- Include comprehensive context for all concepts
- Provide complete code examples with explanations
- Use clear, unambiguous language
- Include error handling and edge cases
- Add metadata about dependencies and versions
- Structure information hierarchically

### Quality Assurance Checklist

Before finalizing documentation:
- [ ] All code examples are tested and functional
- [ ] Links are verified and working
- [ ] Spelling and grammar are correct
- [ ] Formatting is consistent throughout
- [ ] Information is up-to-date and accurate
- [ ] Documentation serves both human and LLM readers effectively

### Input Processing Instructions

When given a project or codebase:
1. **Analyze**: Understand the project's purpose, architecture, and key features
2. **Structure**: Organize information logically and hierarchically
3. **Generate**: Create both README.md and llms.txt files
4. **Optimize**: Ensure content is optimized for both human readability and LLM consumption
5. **Validate**: Review for completeness and accuracy

### Output Format

Always provide:
1. **README.md**: Human-friendly documentation following markdown standards
2. **llms.txt**: LLM-optimized documentation following the llms.txt format
3. **Summary**: Brief explanation of the generated documentation structure

### Additional Considerations

- Adapt tone and technical depth based on the project's target audience
- Include relevant badges and shields for professional appearance
- Ensure accessibility compliance (alt text for images, proper heading hierarchy)
- Consider internationalization if the project has global reach
- Include performance benchmarks or metrics when relevant
- Add security considerations and best practices when applicable

Remember: Great documentation is not just about what you include, but how you present it. Prioritize clarity, completeness, and usability for both human developers and AI systems.
"""

def main():
    load_dotenv()
    # 인자 파싱: 프로젝트 경로, 모델 이름, device
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", type=str, default="/project", help="Project directory path")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu, mps, cuda)")
    args = parser.parse_args()
    project_path = args.project_path
    model_name = args.model_name
    device = args.device

    # HuggingFace 로컬 LLM 사용 (모델명/디바이스 인자화)
    llm = HuggingFaceBackend(model_name=model_name, device=device)

    # LangGraph Agent 예시 - 로컬 LLM을 agent에 연결
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage

    class LocalChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "local-phi3"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            prompt = "\n".join([
                m.content for m in messages if hasattr(m, "content") and isinstance(m.content, str)
            ])
            response = llm.generate(prompt)
            return AIMessage(content=response)

        def invoke(self, messages, config=None, **kwargs):
            # messages가 dict 형태로 전달될 수 있으므로 처리
            if isinstance(messages, dict) and "messages" in messages:
                messages = messages["messages"]
            # messages가 리스트가 아닌 경우 처리
            if not isinstance(messages, list):
                messages = [messages]
            # BaseMessage 객체로 변환
            from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
            if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
                # dict 형태의 메시지를 BaseMessage로 변환
                converted_messages = []
                for msg in messages:
                    if msg.get("role") == "user":
                        converted_messages.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        converted_messages.append(AIMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "system":
                        converted_messages.append(SystemMessage(content=msg.get("content", "")))
                messages = converted_messages
            # 타입 캐스팅으로 타입 에러 해결
            return self._generate(messages, **kwargs)  # type: ignore

    files = read_all_files_relative(project_path)
    print(len(files))
    

    
    summaries = None

    # summaries = {'app.py': "The code defines a comprehensive application for recommending ETFs based on user profiles and queries. It includes various components such as environmental setup, data retrieval, state definitions, user profile analysis, SQL query generation, candidate ETF search, ranking, and explanation. The application uses the LangChain framework to handle the logic and interactions, and it is integrated with the OpenAI API for embeddings and large language models. The final output is a markdown-formatted explanation of the recommended ETFs.\n\nThe main function processes user messages and uses a state graph to orchestrate the flow of the application. The state graph includes the following steps:\n1. Analyze the user's profile to generate an investment profile.\n2. Generate an SQL query based on the user's question and investment profile.\n3. Execute the SQL query to retrieve candidate ETFs.\n4. Rank the ETFs based on the investment profile.\n5. Generate a detailed explanation of the recommendations.\n\nThe application is designed to be modular and extensible, allowing for the addition of new features and integrations in the future.\n```\n\n### Summary\n\nThe provided `app.py` file is a comprehensive application for recommending ETFs based on user profiles and queries. It includes various components such as environmental setup, data retrieval, state definitions, user profile analysis, SQL query generation, candidate ETF search, ranking, and explanation. The application uses the LangChain framework to handle the logic and interactions, and it is integrated with the OpenAI API for embeddings and large language models. The final output is a markdown-formatted explanation of the recommended ETFs.\n\nThe main function processes user messages and uses a state graph to orchestrate the flow of the application. The state graph includes the following steps:\n1. Analyze the user's profile to generate an investment profile.\n2. Generate an SQL query based on the user's question and investment profile.\n3. Execute the SQL query to retrieve candidate ETFs.\n4. Rank the ETFs based on the investment profile.\n5. Generate a detailed explanation of the recommendations.\n\nThe application is designed to be modular and extensible, allowing for the addition of new features and integrations in the future.\n``` \n\n### Key Components and Features\n\n1. **Environmental Setup and Database Connection**:\n   - Loads environment variables from a `.env` file.\n   - Connects to an SQLite database named `etf_database.db` using `SQLDatabase`.\n\n2. **Data Retrieval and Preprocessing**:\n   - Retrieves lists of ETFs, fund managers, and underlying assets from the database.", 'main.py': 'The code is a simple script that prints "Hello from etf-bot!" when run. It uses the if __name__ == "__main__": guard to ensure that the main() function is called only when the script is executed directly, not when it\'s imported as a module.\n\nThe code is straightforward and does not contain any errors. It\'s a simple example of a Python script that prints a greeting.\nThe code is a simple script that prints "Hello from etf-bot!" when run. It uses the if __name__ == "__main__": guard to ensure that the main() function is called only when the script is executed directly, not when it\'s imported as a module.\n\nThe code is straightforward and does not contain any errors. It\'s a simple example of a Python script that prints a greeting.\nThe code is a simple script that prints "Hello from etf-bot!" when run. It uses the if __name__ == "__main__": guard to ensure that the main() function is called only when the script is executed directly, not when it\'s imported as a module.\n\nThe code is straightforward and does not contain any errors. It\'s a simple example of a Python script that prints a greeting.\nThe code is a simple script that prints "Hello from etf-bot!" when run. It uses the if __name__ == "__main__": guard to ensure that the main() function is called only when the script is executed directly, not when it\'s imported as a module.\n\nThe code is a simple script that prints "Hello from etf-bot!" when run. It uses the if __name__ == "__main__": guard to ensure that the main() function is called only when the script is executed directly, not when it\'s imported as a module.\n\nThe code is a simple script that prints "Hello from etf-bot!" when run. It uses the if __name__ == "__main__": guard to ensure that the main() function is called only when the script is executed directly, not when it\'s imported as a module.\n\nThe code is a simple script that prints "Hello from etf-bot!" when run. It uses the if __name__ == "__main__": guard to ensure that the main() function is called only when the script is executed directly, not when it\'s imported as a module.\n\nThe code is a simple script that prints "Hello from etf-bot!" when run. It uses the if __name__ == "__main__": guard to ensure that the main() function', '.venv/bin/activate_this.py': 'sys.executable = os.path.join(base, "bin", "python")\n\n# Add the virtual environment\'s site-packages to the host python\'s sys.path\n# The previous code that did this was in the runpy.run_path() function\n# which was in the virtualenv\'s __init__.py\n# So we need to re-add it to ensure that the virtualenv\'s packages are available\n# when the virtualenv is activated\n# Also, this should be done for the new virtualenv, since the virtualenv\'s\n# site-packages is not added to sys.path by default.\n\nif sys.version_info >= (3, 12):\n    # the standard library has a new version of sys which is not in the\n    # virtualenv\'s site-packages, and we need to make sure that the virtualenv\n    # is compatible with that\n    sys.meta_path = [x for x in sys.meta_path if not (isinstance(x, type) and x.__name__ == "PyStringProxy")]\nelse:\n    # The virtualenv\'s site-packages is not in the standard library\n    # so the Python version is not 3.12, and we can just re-add the site-packages\n    pass\n\n# Make sure that the virtual environment is activated\n# by setting the prompt and the environment variable\nos.environ["PS1"] = f"({os.path.basename(base)}){os.environ.get(\'PS1\', \'\')}"\nos.environ["PS2"] = f"({os.path.basename(base)}){os.environ.get(\'PS2\', \'\')}"\n\n# The following code was originally in the runpy.run_path() function\n# which was in the virtualenv\'s __init__.py\n# We need to make sure that the virtualenv is activated\n# by setting the prompt and the environment variable\n# Also, we need to make sure that the virtualenv is activated\n# by setting the prompt and the environment variable\n# So we need to add the following code here\n\nif sys.version_info >= (3, 12):\n    sys.meta_path = [x for x in sys.meta_path if not (isinstance(x, type) and x.__name__ == "PyStringProxy")]\nelse:\n    pass\n\nif sys.version_info >= (3, 12):\n    # The virtualenv\'s site-packages is not in the standard library\n    # So we need to add the virtualenv\'s site-packages to sys.path\n    # which is the same as the code above', '.venv/lib/python3.12/site-packages/_virtualenv.py': "The code in this file is part of a virtual environment setup and is responsible for patching the distutils and setuptools modules to ensure that they do not use global configuration files that could interfere with the virtual environment's setup. Here's a breakdown of the key components:\n\n- The file imports the necessary modules and defines a constant VIRTUALENV_PATCH_FILE which is the path to the current file.\n- The patch_dist function is used to modify the distutils' parse_config_files method. It ensures that certain configuration options are set to the virtual environment's paths and removes any global configuration options that could be misused.\n- The _Finder class is a meta path finder that is used to patch the imported distutils modules. It checks for the presence of certain module names in the _DISTUTILS_PATCH list and replaces their configuration methods with the patch_dist function.\n- The code inserts the _Finder into the sys.meta_path to ensure that it is used during module loading.\n\nThe overall goal of this code is to prevent the use of global configuration files by distutils and setuptools, thereby ensuring that the virtual environment's paths are correctly set and that the installation process is consistent and isolated from the system's global configuration.\n\nThe code may have some issues, such as the use of the threading module in a non-thread-safe way, and the possibility of the _Finder class being imported prematurely. It also uses the sys.meta_path to modify the import process, which can be a point of instability.\nThe code in this file is part of a virtual environment setup and is responsible for patching the distutils and setuptools modules to ensure that they do not use global configuration files that could interfere with the virtual environment's setup. Here's a detailed breakdown of the key components:\n\n1. **Import Statements**: The file imports necessary modules and defines a constant `VIRTUALENV_PATCH_FILE` which is the path to the current file.\n\n2. **patch_dist Function**:\n   - This function modifies the `parse_config_files` method of `dist.Distribution` to handle configuration files.\n   - It ensures that certain configuration options (like `prefix`) are set to the virtual environment's paths.\n   - It removes any global configuration options that could be misused by the virtual environment.\n\n3. **_Finder Class**:\n   - This is a meta path finder that allows patching the imported `distutils` and `setuptools` modules.\n   - It checks if the module name is in the `_DISTUTILS_PATCH` list and replaces its configuration methods"}
    summaries = {'main.py': "The code file `main.py` is a comprehensive script that includes various tools for interacting with the file system, generating documentation, and utilizing a local Hugging Face model for text generation. The script is designed to be used in a project context, where it processes files, generates documentation, and integrates with a local LLM backend. The code is structured to handle both file operations and documentation generation, leveraging the LangGraph framework for agent-based interaction.\n\nThe main functionalities of the script include:\n\n1. **File System Tools**: The script provides tools for listing directories, creating directories, reading and writing files, and reading all files in a project directory. These tools are implemented using standard Python libraries such as `os` and `pathlib`.\n\n2. **Documentation Generation**: The script includes a `read_all_files_relative` function that recursively reads all files in a project directory, and a `main` function that processes these files to generate a `README.md` and `llms.txt` documents. The documentation is generated based on the content of the files, following specific formatting and structure guidelines.\n\n3. **LLM Integration**: The script uses a local Hugging Face model (`microsoft/Phi-3-mini-4k-instruct`) for text generation. It is integrated with the LangGraph framework to create an agent that can handle complex tasks, such as generating documentation based on the project's file content.\n\n4. **Command Line Arguments**: The script accepts command line arguments to specify the project directory, model name, and device for inference. This allows the user to customize the behavior of the script according to their specific needs.\n\n5. **Output Handling**: The script generates and prints the output of the documentation generation process, including the `README.md` and `llms.txt` files, along with a summary of the generated content.\n\nThe code is designed to be modular and extensible, with the ability to integrate with different LLM backends (such as OpenAI) and to handle various file operations. The use of the LangGraph framework allows for a more sophisticated and interactive approach to documentation generation, enabling the script to handle complex workflows and state management.\n```\n\n### Summary of the Code\n\nThe `main.py` file is a comprehensive script that includes tools for file system operations, documentation generation, and integration with a local Hugging Face model for text generation. It processes files in a project directory, generates `README.md` and `llms.txt` documentation files, and uses the LangGraph framework to create an agent for complex tasks like documentation"}


    if summaries is None:
        print("start summaries")
        summaries = {}
        for rel_path, code in files.items():
            if rel_path.endswith(".py") and not rel_path.startswith("."):
                print(f"===== {rel_path} =====\n")
                prompt = f"Summarize the following code file ({rel_path}):\n\n{code}"
                summary = llm.generate(prompt)
                summaries[rel_path] = summary
                print(f"===== {rel_path} =====\n{summary}\n")
    
    print(summaries)

    model = LocalChatModel()
    agent = create_react_agent(model, tools)
    user_prompt = f"Check the files of project and generate a README.md in 512 tokens. \n\nThere are project files: {files.keys()}\n\n There are summaries of the files: {summaries}"
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    })
    pprint.pprint(result)

    print("==========README.md==========")
    print(result["messages"][-1].content)

    write_file(project_path + "README.md", result["messages"][-1].content)
if __name__ == "__main__":
    main()
