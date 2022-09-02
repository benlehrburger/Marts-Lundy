# Introduction

Early in 2020, Marts & Lundy asked me to help answer the question: how can we predict major donors in higher education? Over the past eight months, I’ve developed regression analyses and neural networks to work towards an answer. I am proud to say that the project was successful.

The insights that I made would not have been possible without the Advizor dataset regarding philanthropy in higher education generously donated for research purposes by Doug Cogswell. This dataset included 65,000+ samples of donors with 23 features each, amassing to more than 1.5M+ data points. That data is sensitive and is not to be shared.

The neural network detailed below performs at 76% accuracy overall and predicts 95% of candidates' donations within $250 of their actual gift. The model classifies donors into one of five classes, indicating the dollar amount that they are likely to give over the next five years. The regression analyses I performed also returned the five most predictive features of any donor. These two results in tandem are quite powerful and ought to inform future fundraising efforts in higher education.

# Description

The attached files comprise an analysis of philanthropy in higher education. The important questions that lead this assessment are:

How can universities and their clients know where to direct fundraising resources?
How can we predict major donors?
What specific features of candidates indicate their likelihood of donating?

The data fueling answers to these questions comes from a 1.5M+ point dataset donated by Advizor for research purposes. The original dataset includes 65,000+ donors and their respective features. These datapoints have been subsequently scrubbed, normalized, and trimmed at the base rate.

The code consists of two main programs. First, a linear regression performed on the data to identify which features of donors are most predictive of potential major donors. Second, a neural network classifier that works at 76% accuracy. It accepts 21 features of a donor and classifies the donor into one of five classes:

$0  
$1 - $999  
$1K - $4.9K  
$5K - $24.9K  
$25K+  

A donor's class represents their predicted donations over the next five years. The required input features to the network are:

Number of gifts in the last 5 years  
Number of gifts in the last 6-10 years  
Dollar amount of gifts in the last 10 years  
Lifetime hard commitment  
Lifetime soft commitment  
Total lifetime commitment  
School committees served on in the last 10 years  
Number of school reunions attended since graduating  
Number of sports played in school  
Number of student activities partipated in school  
Whether or not the candidate graduated with a degree  
Age  
Year of graduation  
Whether or not the candidate holds a c-level job  
Whether or not the candidate is a school alumni  
Whether or not the candidate has a school reunion in the next 5 years  
RFM score  
Number of honors graduated with  
Lifetime hard credit  
Lifetime soft credit  
Total lifetime credit  

# Instructions

To run the classifier, navigate to:

> Code/Main/Predict.py  

Run the program. It will prompt you for the 21 necessary inputs in the terminal. Please give all inputs as numbers (ie. no dollar signs, no strings, etc.).

Once you have supplied all the necessary inputs, the classifier will normalize your data and propagate it forwards through the network. In due time, you will see your donor's classification in the terminal.


