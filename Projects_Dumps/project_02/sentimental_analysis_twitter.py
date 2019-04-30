#import

from textblob import textblob
import tweepy

#consumer key and the consumer secret
consumer_key = ''
consumer_secret = ''

#access_token and the access_token_secret
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access.token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Search key')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
