# ETF-Bot

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/etf-bot/actions)  
[![Python Version](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/downloads/release/python-3120/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Description

ETF-Bot is a chat-based assistant designed to recommend Exchange-Traded Funds (ETFs) based on user investment profiles. It leverages advanced language models and LangChain ecosystem tools to analyze user inputs, generate SQL queries over an ETF database, rank ETFs by suitability, and provide detailed investment explanations.

## Features

- User investment profile analysis including risk tolerance, investment horizon, goals, and sector preferences.
- Natural language SQL query generation and execution on SQLite ETF database.
- Semantic search over proper nouns related to ETFs, fund managers, and assets.
- Ranking of ETF candidates based on multiple financial criteria and user profile.
- Generation of detailed ETF recommendation explanations with investment strategies and risk considerations.
- Web interface using Gradio for easy user interaction.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/etf-bot.git
   cd etf-bot
   ```

2. Create and activate a Python 3.12+ virtual environment:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows use .\.venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Setup environment variables:

   - Create a `.env` file if it doesn't exist and add your OpenAI API key or other required keys.

5. Run the application:

   ```bash
   python app.py
   ```

## Usage

Use the web chat interface launched by the app to enter questions about ETF recommendations. Example:

```
20대 후반의 대학생입니다.
월 50만원 정도를 1년 이상 장기 투자하고 싶고,
보수적 성향이며 ESG 요소도 고려하고 싶습니다.
적절한 ETF를 추천해주세요.
```

The bot will respond with ranked ETF recommendations and detailed explanations.

## API Documentation

This project primarily uses LangChain with OpenAI models and a SQLite database; it does not expose REST API endpoints.

## Contributing

Contributions are welcome! Please submit issues and pull requests via GitHub.

- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

Please follow PEP8 style guidelines and write clear commit messages.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

- v0.1.0 Initial release with core investment profile analysis and ETF recommendations.

---

*This README was autogenerated based on the project files.*
