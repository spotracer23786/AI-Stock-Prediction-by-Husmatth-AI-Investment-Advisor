from newsapi import NewsApiClient
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

newsapi = NewsApiClient(
    api_key=API_KEY
)

def get_sentiment(stock_name):

    news = newsapi.get_everything(
        q=stock_name,
        language="en",
        sort_by="publishedAt",
        page_size=10
    )

    articles = news["articles"]

    if len(articles) == 0:
        return "NO NEWS FOUND"

    sentiment_score = 0

    for article in articles:

        title = article["title"]

        analysis = TextBlob(title)

        sentiment_score += analysis.sentiment.polarity

    avg_score = sentiment_score / len(articles)

    if avg_score > 0.1:
        return "BULLISH 🟢"

    elif avg_score < -0.1:
        return "BEARISH 🔴"

    else:
        return "NEUTRAL 🟡"


result = get_sentiment("Apple")

print("Sentiment:", result)