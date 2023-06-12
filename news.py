import requests
from bs4 import BeautifulSoup
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Function to extract article text from HTML
def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text().strip() for p in paragraphs])
    return text

# Function to generate a short summary using a pre-trained model
def generate_summary(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1000, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Main function
def get_article_summary(article_link):
    # Fetch the article HTML
    response = requests.get(article_link)
    html = response.text

    # Extract the article text from HTML
    article_text = extract_text_from_html(html)

    # Generate a summary of the article
    summary = generate_summary(article_text)

    return summary

def convert(article_link):
    summary = get_article_summary(article_link)
    return summary

