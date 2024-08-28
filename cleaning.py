import re
from bs4 import BeautifulSoup
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text, remove_stopwords=False):
    # Check if the text resembles HTML content
    if '<' in text and '>' in text:
        text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Expand contractions : don't will become do not
    text = contractions.fix(text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Optionally remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]
    
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Rejoin words into a single string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Load CSV file
df = pd.read_csv('extracted_data.csv')

# Apply the clean_text function to the 'subreddit' column
df['cleaned_subreddit'] = df['subreddit'].apply(lambda x: clean_text(x))

# Save the cleaned data to a new CSV file (optional)
df.to_csv('cleaned_data.csv', index=False)

# Display the cleaned data
print(df[['subreddit', 'cleaned_subreddit']])

