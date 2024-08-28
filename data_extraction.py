import praw
import pandas as pd

# Reddit API credentials
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
user_agent = 'YOUR_USER_AGENT'

# Initialize the Reddit instance
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Function to scrape the subreddit
def scrape_subreddit(subreddit_name, limit=1000):
    subreddit = reddit.subreddit(subreddit_name)
    
    # Fetching the hot posts (you can also use .new(), .top(), etc.)
    threads = subreddit.hot(limit=limit)
    
    # List to store the thread details
    threads_list = []
    
    for thread in threads:
        threads_list.append([thread.title, thread.selftext,thread.subreddit])
    
    # Convert list to DataFrame
    threads_df = pd.DataFrame(threads_list, columns=['title', 'selftext', 'subreddit'])
    
    return threads_df

# Scrape the subreddit
subreddit_name = 'PTSD'
threads_df = scrape_subreddit(subreddit_name, limit=1000)

# Save the DataFrame to a CSV file
threads_df.to_csv(f'{subreddit_name}_threads.csv', index=False)

print(f'Scraped {len(threads_df)} threads from r/{subreddit_name}')
