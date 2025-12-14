# 🤖 Enhanced Q&A Chatbot with OpenAI

A simple and interactive **Q&A Chatbot** built using **Streamlit**, **LangChain**, and **OpenAI**.  
This project demonstrates how to build a clean, secure, and production-ready GenAI application.

---

## ✨ Features

- 💬 Ask natural language questions and get instant responses  
- 🤖 Powered by OpenAI models via LangChain  
- 🎛 Control creativity using **Temperature**  
- 📏 Control response length using **Max Tokens**  
- 🔍 LangSmith tracing enabled for observability  
- 🖥️ Clean and minimal Streamlit UI  

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **OpenAI**
- **LangSmith**
- **python-dotenv**

---

## 📂 Project Structure

├── app.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore


⚠️ Important Notes

GPT-5 models are not yet supported by LangChain’s ChatOpenAI abstraction.

For stable usage, prefer gpt-4o or gpt-4o-mini.

Never commit your .env file or API keys to GitHub.
