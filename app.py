import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATABASE = None

embed_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="I'm a bot, please talk to me!")


async def load(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loader = TextLoader('db_test.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    global DATABASE
    DATABASE = FAISS.from_documents(docs, embedding=embed_model)
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Document loaded!")


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs = DATABASE.similarity_search(update.message.text, k=4)
    chain = load_qa_chain(llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY), 
                          chain_type="stuff")
    results = chain({
        'input_documents': docs,
        "question": update.message.text
    },
        return_only_outputs=True)
    text = results['output_text']
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


if __name__ == "__main__":
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('load', load))
    application.add_handler(CommandHandler('query', query))
    application.run_polling()
