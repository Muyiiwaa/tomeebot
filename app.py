import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

DATABASE = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="I'm a bot, please talk to me!")


async def load(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loader = TextLoader('db_test.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    global DATABASE
    DATABASE = FAISS.from_documents(docs, OpenAIEmbeddings())
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Document loaded!")


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs = DATABASE.similarity_search(update.message.text, k=4)
    chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
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
