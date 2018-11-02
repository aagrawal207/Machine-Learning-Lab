
# coding: utf-8

# In[21]:


import nltk
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import random


# In[22]:


def preprocessing(line, negative_contruct:list, smile_positive:list, smile_negative:list, ps:'Porter Stemmer', stop_words:list):
    line = re.sub(r'http\S+', '', line)  # remove URL
    line = re.sub('@[\w_]+', '', line)   # remove USER_MENTIONS
    line = re.sub('#[\w]+', '', line)   # remove hashtags
    
    line = line.lower().split()
    
    # replace negative constructs with not
    # removing stop words
    # replacing emoticons
    for i, word in enumerate(line):
        if line[i] in negative_construct:
            line[i] = "not"
        elif line[i] in stop_words:
            line[i] = ''
        elif line[i] in smile_positive:
            line[i] = "smile_positive"
        elif line[i] in smile_negative:
            line[i] = "smile_negative"
    line = ' '.join(line)
    
    # remove punctuations
    translator = str.maketrans('', '', string.punctuation)
    line = line.translate(translator)
    line = ' '.join(line.split())

    # stemming words
    line = line.split()
    for i, word in enumerate(line):     # replace negative constructs with not
        if line[i] == "smilepositive":
            line[i] = "smile_positive"
            continue
        if line[i] == "smilenegative":
            line[i] = "smile_negative"
            continue
        line[i] = ps.stem(line[i])      # stemming
    line = ' '.join(line)
    
    return line


# In[23]:


file ='./Dataset.txt'
negative_construct = [ "can't", "wouldn't", "wasn't", "hadn't", "never", "won't"]

smile_pos = """:‑) :-] :-3 :-> 8-) :-} :o) :c) :^) =] =) :) :] :3 :> 8) :} :‑D :D 8‑D 8D x‑D xD
X‑D XD =D =3 B^D :-)) :'‑) :') :‑O :O :‑o :o :-0 8‑0 >:O :-* :* :× ;‑) ;) *-) *)
;‑] ;] ;^) :‑, ;D :‑P :P X‑P XP x‑p xp :‑p :p :‑Þ :‑Þ :‑þ :þ :Þ :Þ :‑b :b d: =p
>:P O:‑) O:) 0:‑3 0:3 0:‑) 0:) 0;^) |;‑) :‑J #‑) %‑) %) <3 @};- @}->--
@}‑;‑'‑‑‑ @>‑‑>‑‑""".split()

smile_neg = """:‑( :( :‑c :c :‑< :< :‑[ :[ :-|| >:[ :{ :@ >:( :'‑( :'( D‑': D:< D: D8 D; D= DX :‑/
:/ :‑. >:\ :L =L :S :‑| :| :‑X :X :‑# :# :‑& :& >:‑) >:) }:‑) }:) 3:‑) 3:) >;) ',:-l
',:-| >_> <_< <\3 </3
""".split()

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))


# In[24]:


lines = []
unique_words = []
with open(file, encoding='utf-8') as f:
    for line in f:
        line = preprocessing(line, negative_construct, smile_pos, smile_neg, ps, stop_words)
        unique_words.extend(line.split())
        lines.append(line)

unique_words = set(unique_words)
d = len(unique_words)


# In[25]:


def get_train_test(lines: 'list of sentences', percent):
    data = []
    for line in lines:
        outcome = int(line.split()[0])
        sentence = ' '.join(line.split()[1:])
        temp = [sentence, outcome]
        data.append(temp)
    
    random.shuffle(data)
    split_index = int(percent * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    return train_data, test_data       


# In[26]:


train_data, test_data = get_train_test(lines, 0.75)


# In[27]:


def generate_probabilities(dataset:'list[0] = sentence, list[1]= outcome', d: 'total unique words in train data'):
    prob = defaultdict(int)
    outcome_set = set()
    # first store the counts of (word, outcome)
    for sentence, outcome in dataset:
        outcome_set.add(outcome)
        prob[(outcome, None)] += 1 # (outcome, None) gives count of the outcome
        for word in sentence.split():
            tup = (word, outcome)
            prob[tup] += 1
            
    # convert the counts to probabilities
    for tup, count in prob.items():
        if tup[1] is not None:
            prob[tup] = (prob[tup] + 1) / (prob[(tup[1], None)] + d) # tup[1] is the outcome and (outcome, None) has count of the outcome
    # convert count of outcome to prob
    for outcome in outcome_set:
        prob[(outcome, None)] = prob[(outcome, None)] / len(dataset)
    return (prob, outcome_set)


# In[28]:


probabilities, possible_outcomes = generate_probabilities(train_data, d)


# In[29]:


def bayes(probabilities:'dictionary of tuple to prob', sentence, possible_outcomes):
    best_outcome = -1
    best_prob = -1
    for outcome in possible_outcomes:
        cur_prob = 1
        for word in sentence.split():
            tup = (word, outcome)
            cur_prob *= probabilities[tup]
        cur_prob *= probabilities[(outcome, None)]
        if cur_prob > best_prob:
            best_prob = cur_prob
            best_outcome = outcome
    return best_outcome
    


# In[30]:


correct_and_predicted = [] # list which stores correct outcome and predicted outcome for accuracy, f1 etc. temp[0] is correct. temp[1] is predicted
for sentence, outcome in test_data:
    predict = bayes(probabilities, sentence, possible_outcomes)
    temp = [outcome, predict]
    correct_and_predicted.append(temp)


# In[31]:


true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for given, predicted in correct_and_predicted:
    if given == 1 and predicted == 1:
        true_positive += 1
    elif given == 1 and predicted == 0:
        false_negative += 1
    elif given == 0 and predicted == 0:
        true_negative += 1
    elif given == 0 and predicted == 1:
        false_positive += 1

    
precision = true_positive / (true_positive + false_positive) # true_positive out of all the predicted positive
recall = true_positive / (true_positive + false_negative) # ture_positive out of all the actual positive
f1_score = (2 * precision * recall) / (precision + recall)
accuracy = (true_positive + true_negative) / len(correct_and_predicted) # total correct predicted
print("Precision =", precision)
print("Recall =", recall)
print("F1 Score =", f1_score)
print("Accuracy = ", accuracy)

