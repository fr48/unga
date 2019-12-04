# # # # # # # # # # # # # # # # # # # #
# Classifying the Digital Deciders:
# The Choice Between an Open Internet and a Sovereign One
# Fahmida Y Rashid (fr48)
# Part 3: Using a Classifier
# # # # # # # # # #

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

a = "I hate this"
b= " Zit city."
c= "I wanted to love it but it just wasn't meant for me"
d= "You killed it. FIVE STARS"
e= "I hated it and then I realized I was using it wrong."

reviews = [a,b,c,d,e]

#import nltk
#nltk.download('movie_reviews')

for r in reviews:
    NB = TextBlob(r,analyzer=NaiveBayesAnalyzer()).sentiment
    P = TextBlob(r).sentiment
    print(r,"\n", "bayes=", NB, "\npattern=", P)

from textblob.classifiers import NaiveBayesClassifier

# train set is a list of strings [(),(),()]

train=[
  ('Amazing blend and it is easy to apply.. The product stayed on my face and did not cake up.','over54'),
  ('I’m in love with it ❤️ it’s has a great coverage','25-34'),
  ('Everyone raves about this, but kinda looks cakey on me. Pretty much like CC cream without the benefits; good coverage tho','13-17'),
  ('I have finally found a hydrating foundation that covers up the redness on my face. It has such an easy application.','35-44'),
  ('My skin looks hydrated and bouncy all day long. And it covers all day. My skin never looks dry or flaky.','35-44'),
  ('This Foundation did not sit well on top of my moisturizer and regular sunscreen. It clung to my dry patches it did not sink in to my skin.','35-44'),
  ('I waited to write a review until I had used this foundation multiple times, and I am very very disappointed with it.','13-17'),
  ('LOVE the primer but the foundation looks blotchy, does not stay on, and is extremely hard to blend regardless of my application tool.','13-17'),
  ('I think this will be a perfect winter foundation as my skin tend to dry out a bit.','25-34'),
  ('I really wanted this foundation to work out. Rihanna is really doing a good job with her lines buuuut this fell a little short.','over54'),
  ('Very little coverage and left my face super oily. Guess this foundation does not work for me. will be returning','25-34'),
  ('I LOVE this foundation. The shade is perfect, it feels so comfortable, and wears very well!','25-34'),
]

test=[
    ('I love her other foundation so I wanted to try this one and I am glad I did this one is more hydrating so this is great for good season perfect for dry skin.','25-34'),
    ('Very very unimpressed still. Probably a throw away foundation for me.','13-17'),
    ('I could not get this foundation to set. I would slightly touch it and they would be so much of it under my nails, I could not handle it anymore and had to return it.','over54')
 ]

my_classifier = NaiveBayesClassifier(train)
my_classifier.classify("thanks riri for this amazing foundation")

prob_dist = my_classifier.prob_classify('Fenty foundation is now the only foundation I use')
print(prob_dist.max())
print(round(prob_dist.prob('13-17'), 2))
print(round(prob_dist.prob('25-34'),2))
print(round(prob_dist.prob('over54'),2))

accuracy=my_classifier.accuracy(test)
print(accuracy)
informative=my_classifier.show_informative_features(5)

#alt_classifier = NaiveBayesClassifier([x for x in train if (train???)])
#    eve indices
#alt_accuracy = alt.classifier.accuracy(test)