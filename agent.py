# Import necessary libraries
import os
import requests
import fnmatch
import argparse
import base64
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get GitHub token from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Raise an exception if the token is not set
if GITHUB_TOKEN is None:
    raise Exception("Please set the GITHUB_TOKEN in the .env file.")

# Function to parse GitHub URL and get owner and repo name
def parse_github_url(url):
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo

# Function to get all files from a GitHub repository
def get_files_from_github_repo(owner, repo, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")

# Function to fetch contents of all Markdown files in a repository
def fetch_md_contents(files):
    md_contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], "*.md"):
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode('utf-8')
                print("Fetching Content from ", file['path'])
                md_contents.append(Document(page_content=decoded_content, metadata={"source": file['path']}))
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return md_contents

# Function to split the contents of Markdown files into smaller chunks
def get_source_chunks(files):
    print("In get_source_chunks ...")
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in fetch_md_contents(files):
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadate=source.metadata))
    return source_chunks

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch all *.md files from a GitHub repository.")
    parser.add_argument("url", help="GitHub repository URL")
    args = parser.parse_args()

    # Get owner and repo name from the GitHub URL
    GITHUB_OWNER, GITHUB_REPO = parse_github_url(args.url)
    
    # Get all files from the repository
    all_files = get_files_from_github_repo(GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)

    # Set the path for the Chroma database
    CHROMA_DB_PATH = f'./chroma/{os.path.basename(GITHUB_REPO)}'

    chroma_db = None

    # If the Chroma database does not exist, create it
    if not os.path.exists(CHROMA_DB_PATH):
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        source_chunks = get_source_chunks(all_files)
        chroma_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        chroma_db.persist()
    # If the Chroma database exists, load it
    else:
        print(f'Loading Chroma DB from {CHROMA_DB_PATH} ... ')
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

    # Load the question answering chain
    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=chroma_db.as_retriever())

    # Loop to ask questions and get answers
    while True:
        print('\n\n\033[31m' + 'Ask a question' + '\033[m')
        user_input = input()
        print('\033[31m' + qa.run(user_input) + '\033[m')

# Call the main function if this file is run directly
if __name__ == "__main__":
    main()