# Twitter-sentiment-analysis-using-Python-and-NLTK
This is a practice project which is from this website: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
This post describes the implementation of sentiment analysis of tweets using Python and the natural language toolkit NLTK. The post also describes the internals of NLTK related to this implementation.

Background
The purpose of the implementation is to be able to automatically classify a tweet as a positive or negative tweet sentiment wise.

The classifier needs to be trained and to do that, we need a list of manually classified tweets. Let’s start with 5 positive tweets and 5 negative tweets.

Positive tweets:

I love this car.
This view is amazing.
I feel great this morning.
I am so excited about the concert.
He is my best friend.
Negative tweets:

I do not like this car.
This view is horrible.
I feel tired this morning.
I am not looking forward to the concert.
He is my enemy.
In the full implementation, I use about 600 positive tweets and 600 negative tweets to train the classifier. I store those tweets in a Redis DB. Even with those numbers, it is quite a small sample and you should use a much larger set if you want good results.

Next is a test set so we can assess the exactitude of the trained classifier.

Test tweets:

I feel happy this morning. positive.
Larry is my friend. positive.
I do not like that man. negative.
My house is not great. negative.
Your song is annoying. negative.
Implementation
The following list contains the positive tweets:

1
pos_tweets = [('I love this car', 'positive'),
2
              ('This view is amazing', 'positive'),
3
              ('I feel great this morning', 'positive'),
4
              ('I am so excited about the concert', 'positive'),
5
              ('He is my best friend', 'positive')]
The following list contains the negative tweets:

1
neg_tweets = [('I do not like this car', 'negative'),
2
              ('This view is horrible', 'negative'),
3
              ('I feel tired this morning', 'negative'),
4
              ('I am not looking forward to the concert', 'negative'),
5
              ('He is my enemy', 'negative')]
We take both of those lists and create a single list of tuples each containing two elements. First element is an array containing the words and second element is the type of sentiment. We get rid of the words smaller than 2 characters and we use lowercase for everything.

1
tweets = []
2
for (words, sentiment) in pos_tweets + neg_tweets:
3
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
4
    tweets.append((words_filtered, sentiment))
The list of tweets now looks like this:

01
tweets = [
02
    (['love', 'this', 'car'], 'positive'),
03
    (['this', 'view', 'amazing'], 'positive'),
04
    (['feel', 'great', 'this', 'morning'], 'positive'),
05
    (['excited', 'about', 'the', 'concert'], 'positive'),
06
    (['best', 'friend'], 'positive'),
07
    (['not', 'like', 'this', 'car'], 'negative'),
08
    (['this', 'view', 'horrible'], 'negative'),
09
    (['feel', 'tired', 'this', 'morning'], 'negative'),
10
    (['not', 'looking', 'forward', 'the', 'concert'], 'negative'),
11
    (['enemy'], 'negative')]
Finally, the list with the test tweets:

1
test_tweets = [
2
    (['feel', 'happy', 'this', 'morning'], 'positive'),
3
    (['larry', 'friend'], 'positive'),
4
    (['not', 'like', 'that', 'man'], 'negative'),
5
    (['house', 'not', 'great'], 'negative'),
6
    (['your', 'song', 'annoying'], 'negative')]
Classifier
The list of word features need to be extracted from the tweets. It is a list with every distinct words ordered by frequency of appearance. We use the following function to get the list plus the two helper functions.

1
word_features = get_word_features(get_words_in_tweets(tweets))
1
def get_words_in_tweets(tweets):
2
    all_words = []
3
    for (words, sentiment) in tweets:
4
      all_words.extend(words)
5
    return all_words
1
def get_word_features(wordlist):
2
    wordlist = nltk.FreqDist(wordlist)
3
    word_features = wordlist.keys()
4
    return word_features
If we take a pick inside the function get_word_features, the variable ‘wordlist’ contains:

01
<FreqDist:
02
    'this': 6,
03
    'car': 2,
04
    'concert': 2,
05
    'feel': 2,
06
    'morning': 2,
07
    'not': 2,
08
    'the': 2,
09
    'view': 2,
10
    'about': 1,
11
    'amazing': 1,
12
    ...
13
>
We end up with the following list of word features:

01
word_features = [
02
    'this',
03
    'car',
04
    'concert',
05
    'feel',
06
    'morning',
07
    'not',
08
    'the',
09
    'view',
10
    'about',
11
    'amazing',
12
    ...
13
]
As you can see, ‘this’ is the most used word in our tweets, followed by ‘car’, followed by ‘concert’…

To create a classifier, we need to decide what features are relevant. To do that, we first need a feature extractor. The one we are going to use returns a dictionary indicating what words are contained in the input passed. Here, the input is the tweet. We use the word features list defined above along with the input to create the dictionary.

1
def extract_features(document):
2
    document_words = set(document)
3
    features = {}
4
    for word in word_features:
5
        features['contains(%s)' % word] = (word in document_words)
6
    return features
As an example, let’s call the feature extractor with the document [‘love’, ‘this’, ‘car’] which is the first positive tweet. We obtain the following dictionary which indicates that the document contains the words: ‘love’, ‘this’ and ‘car’.

01
{'contains(not)': False,
02
 'contains(view)': False,
03
 'contains(best)': False,
04
 'contains(excited)': False,
05
 'contains(morning)': False,
06
 'contains(about)': False,
07
 'contains(horrible)': False,
08
 'contains(like)': False,
09
 'contains(this)': True,
10
 'contains(friend)': False,
11
 'contains(concert)': False,
12
 'contains(feel)': False,
13
 'contains(love)': True,
14
 'contains(looking)': False,
15
 'contains(tired)': False,
16
 'contains(forward)': False,
17
 'contains(car)': True,
18
 'contains(the)': False,
19
 'contains(amazing)': False,
20
 'contains(enemy)': False,
21
 'contains(great)': False}
With our feature extractor, we can apply the features to our classifier using the method apply_features. We pass the feature extractor along with the tweets list defined above.

1
training_set = nltk.classify.apply_features(extract_features, tweets)
The variable ‘training_set’ contains the labeled feature sets. It is a list of tuples which each tuple containing the feature dictionary and the sentiment string for each tweet. The sentiment string is also called ‘label’.

01
[({'contains(not)': False,
02
   ...
03
   'contains(this)': True,
04
   ...
05
   'contains(love)': True,
06
   ...
07
   'contains(car)': True,
08
   ...
09
   'contains(great)': False},
10
  'positive'),
11
 ({'contains(not)': False,
12
   'contains(view)': True,
13
   ...
14
   'contains(this)': True,
15
   ...
16
   'contains(amazing)': True,
17
   ...
18
   'contains(enemy)': False,
19
   'contains(great)': False},
20
  'positive'),
21
  ...]
Now that we have our training set, we can train our classifier.

1
classifier = nltk.NaiveBayesClassifier.train(training_set)
Here is a summary of what we just saw:

Twitter sentiment analysis with Python and NLTK

The Naive Bayes classifier uses the prior probability of each label which is the frequency of each label in the training set, and the contribution from each feature. In our case, the frequency of each label is the same for ‘positive’ and ‘negative’. The word ‘amazing’ appears in 1 of 5 of the positive tweets and none of the negative tweets. This means that the likelihood of the ‘positive’ label will be multiplied by 0.2 when this word is seen as part of the input.

Let’s take a look inside the classifier train method in the source code of the NLTK library. ‘label_probdist’ is the prior probability of each label and ‘feature_probdist’ is the feature/value probability dictionary. Those two probability objects are used to create the classifier.

1
def train(labeled_featuresets, estimator=ELEProbDist):
2
    ...
3
    # Create the P(label) distribution
4
    label_probdist = estimator(label_freqdist)
5
    ...
6
    # Create the P(fval|label, fname) distribution
7
    feature_probdist = {}
8
    ...
9
    return NaiveBayesClassifier(label_probdist, feature_probdist)
In our case, the probability of each label is 0.5 as we can see below. label_probdist is of type ELEProbDist.

1
print label_probdist.prob('positive')
2
0.5
3
print label_probdist.prob('negative')
4
0.5
The feature/value probability dictionary associates expected likelihood estimate to a feature and label. We can see that the probability for the input to be negative is about 0.077 when the input contains the word ‘best’.

1
print feature_probdist
2
{('negative', 'contains(view)'): <ELEProbDist based on 5 samples>,
3
 ('positive', 'contains(excited)'): <ELEProbDist based on 5 samples>,
4
 ('negative', 'contains(best)'): <ELEProbDist based on 5 samples>, ...}
5
print feature_probdist[('negative', 'contains(best)')].prob(True)
6
0.076923076923076927
We can display the most informative features for our classifier using the method show_most_informative_features. Here, we see that if the input does not contain the word ‘not’ then the positive ration is 1.6.

01
print classifier.show_most_informative_features(32)
02
Most Informative Features
03
           contains(not) = False          positi : negati =      1.6 : 1.0
04
         contains(tired) = False          positi : negati =      1.2 : 1.0
05
       contains(excited) = False          negati : positi =      1.2 : 1.0
06
         contains(great) = False          negati : positi =      1.2 : 1.0
07
       contains(looking) = False          positi : negati =      1.2 : 1.0
08
          contains(like) = False          positi : negati =      1.2 : 1.0
09
          contains(love) = False          negati : positi =      1.2 : 1.0
10
       contains(amazing) = False          negati : positi =      1.2 : 1.0
11
         contains(enemy) = False          positi : negati =      1.2 : 1.0
12
         contains(about) = False          negati : positi =      1.2 : 1.0
13
          contains(best) = False          negati : positi =      1.2 : 1.0
14
       contains(forward) = False          positi : negati =      1.2 : 1.0
15
        contains(friend) = False          negati : positi =      1.2 : 1.0
16
      contains(horrible) = False          positi : negati =      1.2 : 1.0
17
...
Classify
Now that we have our classifier initialized, we can try to classify a tweet and see what the sentiment type output is. Our classifier is able to detect that this tweet has a positive sentiment because of the word ‘friend’ which is associated to the positive tweet ‘He is my best friend’.

1
tweet = 'Larry is my friend'
2
print classifier.classify(extract_features(tweet.split()))
3
positive
Let’s take a look at how the classify method works internally in the NLTK library. What we pass to the classify method is the feature set of the tweet we want to analyze. The feature set dictionary indicates that the tweet contains the word ‘friend’.

01
print extract_features(tweet.split())
02
{'contains(not)': False,
03
 'contains(view)': False,
04
 'contains(best)': False,
05
 'contains(excited)': False,
06
 'contains(morning)': False,
07
 'contains(about)': False,
08
 'contains(horrible)': False,
09
 'contains(like)': False,
10
 'contains(this)': False,
11
 'contains(friend)': True,
12
 'contains(concert)': False,
13
 'contains(feel)': False,
14
 'contains(love)': False,
15
 'contains(looking)': False,
16
 'contains(tired)': False,
17
 'contains(forward)': False,
18
 'contains(car)': False,
19
 'contains(the)': False,
20
 'contains(amazing)': False,
21
 'contains(enemy)': False,
22
 'contains(great)': False}
1
def classify(self, featureset):
2
    # Discard any feature names that we've never seen before.
3
    # Find the log probability of each label, given the features.
4
    # Then add in the log probability of features given labels.
5
    # Generate a probability distribution dictionary using the dict logprod
6
    # Return the sample with the greatest probability from the probability
7
    # distribution dictionary
Let’s go through that method using our example. The parameter passed to the method classify is the feature set dictionary we saw above. The first step is to discard any feature names that are not know by the classifier. This step does nothing in our case so the feature set stays the same.

Next step is to find the log probability for each label. The probability of each label (‘positive’ and ‘negative’) is 0.5. The log probability is the log base 2 of that which is -1. We end up with logprod containing the following:

1
{'positive': -1.0, 'negative': -1.0}
The log probability of features given labels is then added to logprod. This means that for each label, we go through the items in the feature set and we add the log probability of each item to logprod[label]. For example, we have the feature name ‘friend’ and the feature value True. Its log probability for the label ‘positive’ in our classifier is -2.12. This value is added to logprod[‘positive’]. We end up with the following logprod dictionary.

1
{'positive': -5.4785441837188511, 'negative': -14.784261334886439}
The probability distribution dictionary of type DictionaryProbDist is generated:

1
DictionaryProbDist(logprob, normalize=True, log=True)
The label with the greatest probability is returned which is ‘positive’. Our classifier finds out that this tweets has a positive sentiment based on the training we did.

Another example is the tweet ‘My house is not great’. The word ‘great’ weights more on the positive side but the word ‘not’ is part of two negative tweets in our training set so the output from the classifier is ‘negative’. Of course, the following tweet: ‘The movie is not bad’ would return ‘negative’ even if it is ‘positive’. Again, a large and well chosen sample will help with the accuracy of the classifier.

Taking the following test tweet ‘Your song is annoying’. The classifier thinks it is positive. The reason is that we don’t have any information on the feature name ‘annoying’. Larger the training sample tweets is, better the classifier will be.

1
tweet = 'Your song is annoying'
2
print classifier.classify(extract_features(tweet.split()))
3
positive
There is an accuracy method we can use to check the quality of our classifier by using our test tweets. We get 0.8 in our case which is high because we picked our test tweets for this article. The key is to have a very large number of manually classified positive and negative tweets.

Voilà. Don’t hesitate to post a comment if you have any feedback.

tags: Python
posted in Uncategorized by Laurent Luce
