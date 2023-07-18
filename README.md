# Personalized Movie Recommender
Code Repository for "Nolan Movie Chatbot" by Nawon Lee, Minh-Huy Le, and Karthik Subbiah.

## Updated 2.0
1. Identify movies without quotation marks and incorrect capitalization
2. Provides recommendations based on the user's mood/ emotions
3. Deals with simple spelling mistakes

<hr>

The goal of this chatbot is to provide movie recommendations to a user. In order to do so, the chatbot will need to:
 
1. Extract and transform a user's input 
2. Predict the sentiment of a user's input about movies 
3. Gather 5 sentiment ratings from the user to predict a new movie 

Here's an example of what an interaction between your chatbot and a human user might look like. 

```
chatbot> Hi! I'm Chatstopher Nolan. I'm going to reccommend the perfect movie. 
        To do that, I need you to tell me about some movies you've liked and disliked. 
        Tell me about a movie you've seen!

user> I really liked Moneyball.
```
The bot gets confused because it only recognizes movies in quotation marks. 

```
chatbot> Sorry, I don't understand. Tell me about a movie that you've seen with the title in quotation marks. 

user> I really liked "Moneyball".

chatbot> You liked "Moneyball". Thank you! Tell me about another movie you have seen.

user> I was completely amazed at how "Rush Hour" was pretty much the most ridiculous thing I've seen in my entire life.
``` 
The sentence above was too hard for the bot to predict sentiment, but the bot recovers. 

```
chatbot > I'm sorry, I'm not quite sure if you liked "Rush Hour". Tell me more about "Rush Hour".

user> I hated "Rush Hour".

chatbot> Ok. You disliked "Rush Hour." There are multiple Rush Hours. 
Did you mean "Rush Hour (1998)", "Rush Hour 2 (2001)", or "Rush Hour 3 (2007)"?

user> "Rush Hour 2"

chatbot> Ok. You meant "Rush Hour 2 (2001)" 
```
Above, the chatbot is able to disambiguate between multiple movies that match "Rush Hour" in its movie database. 

```
chatbot> I want to hear more about movies! Tell me about another movie you have seen.

user> I didn't like "The Notebook".
```
... the bot proceeds to gather sentiment ratings from the user for 5 movies. 

```
chatbot> Thanks! That's enough for me to make a recommendation. 
I suggest you watch "In the Heart of the Sea".

user> Great. Thanks! 

chatbot> Would you like to hear another recommendation? (Or enter :quit if you're done.)

user> :quit

chatbot> Thanks! It was fun chatting with you!  

```

## Data

We have provided the following three datasets which you will use in `chatbot.py`. 

### Sentiment lexicon 

In `data/sentiment.txt`, we provide a sentiment lexicon that you can use to extract sentiment from the input. It consists of 3624 words with their associated sentiment (positive or negative) extracted from Harvard Inquirer (Stone et al. 1966). The lexicon is stored for your convenience in a dictionary/hash map, where the word is the key and the sentiment the value.

### Rotten Tomatoes Reviews

In `data/rotten_tomatoes.pkl`, we provide a subset of the [Rotten Tomatoes dataset from Kaggle.](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data). This dataset includes reviews in the form of text and class labels "fresh" and "rotten". 


### Movie recommendation database 

Your `movie database` consits of two files: 

- `data/movies.txt` 
- `data/ratings.txt`

This data comes from [MovieLens](https://movielens.org/) and consists of a total of 9125 movies rated by 671 users. Feel free to browse this data in a text editor. 

The file `data/ratings.txt` includes a 9125 x 671 utility matrix that contains ratings for users and movies. The ratings range anywhere from 1.0 to 5.0 with increments of 0.5. The code will binarize the ratings as follows:

```
+1 if the user liked the movie (3.0-5.0)
-1 if the user didn’t like the movie (0.5-2.5)
0 if the user did not rate the movie
```

We also provide `data/movies.txt`, a list with 9125 movie titles and their associated movie genres. The movie in the first position in the list corresponds to the movie in the first row of the ratings matrix. An example entry looks like:

```
['Blade Runner (1982)', 'Action|Sci-Fi|Thriller']
```







