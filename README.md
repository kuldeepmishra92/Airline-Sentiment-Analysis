
# ✈️ Airline-Sentiment-Analysis

> **Extracting Passenger Voices**: A sentiment analysis of airline reviews using NLP (VADER, RoBERTa) to surface traveler feedback, trends, and actionable insights.

---

## Table of Contents

1. [Project Objective](#project-objective)
2. [Dataset Description](#dataset-description)
3. [Tools & Techniques](#tools--techniques)
4. [Workflow](#workflow)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Modeling & Sentiment Analysis](#modeling--sentiment-analysis)
7. [Insights & Recommendations](#insights--recommendations)
8. [Challenges & Limitations](#challenges--limitations)
9. [Future Work](#future-work)

---

## Project Objective

* To **decode sentiment** from airline customer reviews, understanding what travelers love, dislike, or feel neutral about.
* To identify **trends**, **key service touchpoints**, and areas with potential for improvement (e.g. food, staff, comfort).
* To provide actionable recommendations to airlines for elevating traveler satisfaction and enhancing operational decisions.

---

## Dataset Description

| Feature           | Details                                                                                                                                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Number of Rows    | \~1,300+ reviews                                                                                                                                                                                                          |
| Number of Columns | \~19 features                                                                                                                                                                                                             |
| Key Columns       | `aircraft`, `traveller_type`, `seat_type`, `route`, `date_flown`, `recommended`, `trip_verified`, `rating`, `seat_comfort`, `cabin_staff_service`, `food_beverages`, `ground_service`, `value_for_money`, `entertainment` |
| What’s Inside     | Review text + quantitative ratings across different dimensions of airline experience                                                                                                                                      |

---

## Tools & Techniques

* **Languages & Libraries**: Python, NLTK, Transformers (for RoBERTa), Pandas, Matplotlib, Seaborn
* **Sentiment Models**:

  * **VADER** — rule-based, fast, suitable for social media / reviews with shorter text
  * **RoBERTa (pretrained + fine-tuned)** — deep learning approach, better for semantic accuracy
* **Text Preprocessing**: Tokenization, Lemmatization, Stop-word removal, Lowercasing, Cleaning punctuation / symbols

---

## Workflow

1. **Data Collection & Cleaning**

   * Import reviews + rating data
   * Clean review text: remove noise (HTML tags, emojis if needed), normalize cases, remove stopwords

2. **Feature Engineering**

   * Derive features such as: `review_length`, `verified_trip_flag`, `traveller_type` categories, etc.
   * Date handling: convert `date_flown` → datetime; maybe derive month / season

3. **Exploratory Data Analysis (EDA)**

   * Rating distributions (overall, by `traveller_type`, by `aircraft`, etc.)
   * Word clouds to visualize commonly used positive / negative words
   * Correlation among service dimensions (food, comfort, staff) vs. overall rating

4. **Sentiment Scoring & Classification**

   * Use **VADER** to get sentiment scores (positive / negative / neutral) + compound score
   * Use **RoBERTa** model to classify reviews, potentially compare results

5. **Comparative Analysis**

   * Compare sentiment scores across categories: `seat_type`, `aircraft`, `traveller_type` etc.
   * Time-based trends: do sentiments improve or worsen over months / seasons?

---

## Exploratory Data Analysis (EDA)

* Word Clouds for frequently used words in *positive* vs *negative* reviews to spot recurring themes
* Distribution plots for ratings across categories (seat comfort, food, staff service)
* Bar charts showing how many reviews are positive, neutral, or negative overall

![Word Cloud / Topic Visualization](https://raw.githubusercontent.com/yourusername/Airline-Sentiment-Analysis/main/wordcloud_positive_negative.png)
*Sample visualization of frequent words in reviews.*

---

## Modeling & Sentiment Analysis

* **VADER Analysis**:

  * Pros: lightweight, fast
  * Cons: might miss context, sarcasm, nuanced expressions

* **RoBERTa Analysis**:

  * Pros: better at capturing context, subtle sentiment
  * Cons: slower, computationally heavier

### Sample Code Snippet (VADER)

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
df['vader_scores'] = df['review_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
df['vader_sentiment'] = df['vader_scores'].apply(lambda score: 
    'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
)
```

### Sample Code Snippet (RoBERTa)

```python
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
df['roberta_sentiment'] = df['review_text'].apply(lambda x: sent_pipeline(x)[0]['label'])
```

---

## Insights & Recommendations

* **Common pain points** from negative reviews:

  * Poor staff behavior / unhelpfulness
  * Food & beverage quality
  * Seat comfort (legroom, cleanliness)

* **Positive highlights**:

  * Friendly cabin staff
  * Timeliness (on-time flights)
  * Verified / accurately described services

* **Recommendations**:

  1. Focus on **staff training** especially in cabin service and customer communication
  2. Improve **seat comfort** (padding, cleanliness, features like armrest, legroom)
  3. Enhance **food & beverage offerings**, perhaps through menu variety or quality standards
  4. Regular **surveys** and feedback loops to track sentiment over time

---

## Challenges & Limitations

* Handling sarcasm, slang, and idiomatic expressions
* Potential bias: reviews may come more from unhappy customers than satisfied ones
* Imbalanced sentiment classes (maybe far more “positive” or “negative”)
* Lack of context in short reviews
* RoBERTa computational cost may restrict scale

---

## Future Work

* Use **topic modeling** (e.g. LDA, BERTopic) to uncover hidden themes beyond basic categories
* Add **time series sentiment tracking** (how sentiment changes over months)
* Incorporate **feedback from external sources** like social media / Twitter
* Build a **dashboard** (using Streamlit / Dash / Power BI / Tableau) for visual interactive insights
* Optionally deploy model as an API so airlines can input new reviews and get sentiment feedback in real-time

---

## 🔗 Useful Resources & References

* [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
* [RoBERTa Paper & Transformers Library](https://huggingface.co/transformers/)
* [NLTK](https://www.nltk.org/) — for text processing
* Research papers / articles on airline review sentiment for thematic inspiration

---

> “Travelers may forget the cost of a ticket, but they’ll never forget how you made them feel.” — *Customer Experience Maxim*
