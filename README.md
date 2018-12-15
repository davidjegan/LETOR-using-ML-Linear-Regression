# LETOR-using-ML-Linear-Regression
The code provides a detailed insight on how Linear regression can be used to solve Learning to Rank problem. We leverage two methodologies., Closed form and Gradient Descent. 



## Learning to Rank
The learning to rank algorithm, gives the closest match between a query and a document. The target is a relevance label of values 0,1 and 2, where 2 denotes the best relation between the query and the document. Though the values in the training target are scalar and discrete (0,1,2), we consider linear regression to obtain continuous value rather than a discrete on, to increase the difference factor between the documents and thereby ranking the pages. 


## Approach
There are three stages in this problem 
1. Create training data
2. Closed form solution
3. Stochastic gradient descent solution

