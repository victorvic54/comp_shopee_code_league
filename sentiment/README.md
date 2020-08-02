## ABOUT THE FINAL DATA

This competition was resetted because the data is leaky and is not randomized, hence we can get the prediction of 100% (this data will be referred as leaky_test_data.csv)

We also included dataset from https://www.kaggle.com/shymammoth/shopee-reviews

Then all of the above data is collected and merge together with the train data that is given by shopee itself and form a temporary final data

This final data is being clean thoroughly by:
```
def removeNumbersAndPunctuations(text):
    text = text.translate(table)
    text = re.sub(r'\d+', '', text)
    return text
```
```
def removeSpaces(text):
    return text.strip()
```

```
def lowerWords(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    
    return " ".join(tokens)
```

```
common_stopwords = ['the', 'is', 'of', 'to', 'for', 'in', 'it', 'you', 'so', 'goods', 'his', 'her', 'we', 'me']

def removeStopWords(text):
    tokens = word_tokenize(text)
    words = [w for w in tokens if not w in common_stopwords]
    
    return " ".join(words)
```

Then finally:
```
df['review'] = df['review'].apply(removeNumbersAndPunctuations)
df['review'] = df['review'].apply(removeSpaces)
df['review'] = df['review'].apply(lowerWords)
df['review'] = df['review'].apply(removeStopWords)

# remove empty data
df = df[df["review"] != ""]

```