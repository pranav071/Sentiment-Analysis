from textblob import TextBlob
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = [
    ("I love this product, it's amazing!", "positive"),
    ("This movie is terrible, I hated it.", "negative"),
    ("The food at that restaurant was fantastic.", "positive"),
    ("I'm really disappointed with the service.", "negative"),
    ("I was disgusted by the room service.", "negative"),
    ("Your creativity shines through in everything you do.", "positive"),
    ("The weather today is gloomy and overcast.", "negative"),
    ("Your willingness to help others is a true asset to any team.", "positive"),
    ("leadership style is inclusive and empowering.", "positive"),
    ("The customer support hotline had me waiting on hold for an unreasonable amount of time.", "negative"),
    ("The public transportation system is unreliable and frequently delayed.", "negative"),
    ("The enthusiasm is infectious and motivates those around you.", "positive"),
    ("The salad I ordered had wilted and soggy lettuce.", "negative"),
    ("Your positive energy lifts the spirits of those around you.", "positive"),
    ("The gym I visited had outdated equipment that's in poor condition.", "negative"),
    ("Your optimism in the face of adversity is truly admirable.", "positive"),
    ("The battery life on this electronic device drains incredibly fast.", "negative"),
    ("The internet connection in this area is slow and unreliable.", "negative"),
    ("Your sense of humor brings joy to any situation.", "positive"),
    ("The Playground was well maintained and clean", "positive")
    
]

texts, labels = zip(*data)

# Convert labels to numerical values (0 for negative, 1 for positive)
label_mapping = {"negative": 0, "positive": 1}
numeric_labels = [label_mapping[label] for label in labels]

# Perform sentiment analysis
predictions = []
sentiment_scores = []
for text in texts:
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment_scores.append(sentiment_score)
    x_values = np.arange(len(data))


    if sentiment_score > 0:
        predictions.append(1)  # Positive sentiment
    else:
        predictions.append(0)  # Negative sentiment

# Calculate accuracy
accuracy = accuracy_score(numeric_labels, predictions)
print("Accuracy:", accuracy)

# Plotting Accuracy
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)  # Set y-axis limits
plt.ylabel("Accuracy")
plt.title("Sentiment Analysis Accuracy")
plt.show()

# Plotting the sentiment polarity scores
plt.bar(x_values, sentiment_scores)
plt.xlabel("data")
plt.ylabel("Sentiment Polarity Score")
plt.title("Sentiment Polarity Scores for data")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()