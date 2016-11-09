# Fact_Checker

# Problem Statement

Given a set of document(s) and a fact, this project decides whether the fact is true or false.

# Approach

1. Multi Document Summarisation - (Sentence Extraction, Redundancy checking, Summary Generation)

2. Build a negation of the fact using NLTK tools(POS tagging and then inverting)

3. Semantic Similarity of the fact and its negation with the summary.

4. Comparision of the scores from Semantic Similarity, and classifying the truthness of the fact.
