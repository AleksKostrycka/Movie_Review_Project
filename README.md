# Movie_Review_Project
## Overview

The purpose of this analysis is to predict the IMDb score of movies based on several parameters, including genre, rating, release date, director, and much more. Each year, hundreds of films are produced worldwide and some are exceptionally successful, and others are surprisingly not. The goal of this project is to predict the success, measured by their IMDb rating, of upcoming films by first analyzing which features contribute heavily to the outcome of the movie, then creating a supervised machine learning model that will predict the IMDb rating. In addition to the supervised machine learning model that we intend to create to predict a movie rating, we are also focusing on several attributes and exploring if there is a correlation between those attributes and the final rating of movies.
The data source for this project was this [dataset](https://www.kaggle.com/danielgrijalvas/movies), which is a compiled tabular dataset of the budget, company, country of origin, director, genre, gross revenue, title, rating, release date, runtime, score, number of votes, stars, and writer, which was all scraped from the IMDb website. We hope to accurately predict the success of prospective movies, as well as determine the main contributing factors of a successful film.

## Outline
What affects movie ratings? How can we predict a movie rating?

In today's world we have so many different ways of watching movies, which also amplifies our choices on what movie to watch. Whether it is in the comfort of our own homes or going to the theater, the decision of what to commit our limited time to is an important one. So how do you narrow down your choices to the movies worth seeing? As a group we will attempt to answer the following questions along with a predictive model for determining movie rating:
* Which studio produces the highest rated movies?
* What genre is the most popular? (the highest rated)
* Does the actor affect the rating of the movie?
* Does the director affect the rating of the movie?

In our project we will present our machine learning model and outcome, describe and show the findings related to the questions above through statistical analysis and data presentation in Tableau.

## Tableau Dashboard:

To be able to quickly describe and summarize out entire dataset we decided to use Tablaeu and its functionality to create the Movie Data Dashboard. This can be found under the link here: [Movie Data Dashboard](https://public.tableau.com/app/profile/aleks.kostrycka/viz/MovieReviewPresentation/Dashboard1?publish=yes). We created visualizations in Tableau to baseline the audience and users of our model on the contents of the data we utilized. We found that creating this dashboard allowed us to touch on key metrics within the data in a clean and succinct way. To summarize the data we called out the number of movies in the data, median, average, min and max as well as created several graphs based on genre, rating, studio. As well as information tables summarizing Director, Actor and Movie Info. 

Figure 1
![This is an image](https://github.com/AleksKostrycka/Movie_Review_Project/blob/readmealeks/Images/Figure%201.png?raw=true)

Another function of the Dashboard is the filterablitly on all the figures seen above. One of our objectives when starting this project was to create a platform that will allow the user to narrow in on a particular movie of interest. Whether the criteria might be their favorite genre, actor, director or studio the user can quickly turn on the correct filters and find the right movie they might be interested in re watching or find something new by their favorite stars/studios/directors. 

For example if a user was interested in seeing an Adam Sandler comedy they might not have seen before, the filters in the image below can be turned on. To show see the highest rated Adam Sandler comedies and we are able to idetify Punch-Drunk Love as a potential movie to watch which has a 7.3 rating, well above the average in the dataset. 

Figure 2
![This is an image](https://github.com/AleksKostrycka/Movie_Review_Project/blob/readmealeks/Images/Figure%202.png?raw=true)

Then we can use the Reset Filter button seen in Figure 1 to clear all filters that we just established. 

Our goal was to answer 2 of our initial questions using the Tableau Dashboard:
* Does the actor affect the rating of the movie?
* Does the director affect the rating of the movie?

To answer our first question "Does the actor affect the movie rating?" We decided to narrow down our set to focus on  one genra, comedy to enable us to give a more percise answer. Once the dataset is filtered on comedy we can see that Steve Martin has 24 comedy movies within the dataset. Looking at the overall results of his movies, the average, min and max we can assume that if he does come out with a new movie this will likely receive the same score as we see in the dataset. The caliber and name recognition of the actor affects the rating. 

Figure 3
![This is an image](https://github.com/AleksKostrycka/Movie_Review_Project/blob/readmealeks/Images/Figure%203.png?raw=true)

The same approach was taken to answer our second question to narrow down comedy directors. Despite his personal life, within the dataset it is clear the Woody Allen has directed the most comedies. 33 Comedies with a very high avg rating of 6.99. In Figure 4 we can present and conclude that, when looking at his work alone, there is a good chance that a movie being directed by Woody Allen will have a higher rating. 

Figure 4
![This is an image](https://github.com/AleksKostrycka/Movie_Review_Project/blob/readmealeks/Images/Figure%204.png?raw=true)

There is a lot of applicability to this dashboard, above we only described a few functions. However when it comes to improvements we can see that more movies would make this dashboard more complete and all encompasing. Also a distinction between fan and critic rating would ehance the information. Another feature would be helpful is a where to watch this movie feature. That would direct the users to the platform where they can find this movie. 

## Summary of Data Preparation
Once we found the data through [dataset](https://www.kaggle.com/danielgrijalvas/movies) our next step was to prep the data set to ensure it was usable and complete. This was done by utilizing Python (Pandas and Sqlalchemy), PostgresSQL, and Jupyter Notebook. 
The ETL process was as follow, extracting the two CSV files, the movie dataset and an inflation dataset. The inflation data set was used to normalize our currency values. The data was then converted into dataframes using Panadas, then loaded into PostgresSQL for transformation. Docker was used to host PostgresSQL, creating a simialr environment across all team members. An ELT was done instead, since transforming the data in PostgresSQL is a faster and efficient way to clean the data compared to Pandas. In the transformation process, we found the budget column had over 2,000 null values. Through various conversation and additional research, we decided that this feature was not entirely necessary as budget was found to have a weak correlation with the outcome of a film. At the end our movie dataset consisted of 5420 rows. 
The inflation dataset was transformed to be able to merge with the movie data set. These prices were adjusted for inflation following this [guide](https://towardsdatascience.com/adjusting-prices-for-inflation-in-pandas-daaaa782cd89).
 
## Statistical Analysis
 
## Machine Learning Model
Three preliminary supervised machine learning models: Linear Regression, Random Forest, and Gradient Boost were prepared to determine the best algorithm for the prediction of IMDb film ratings. The three models were created without any additional data processing to evaluate the metrics and noise of each individual system. The Random Forest model produced the highest r² score, which evaluates the performance of a regression-based machine learning model by measuring the amount of variance in the predictions explained by the linear model. Root mean square error (RMSE), or the square root of the average of the squared value of all error, and mean absolute error (MAE), or the average error, were two other model metrics used to evaluate these models. 

To further improve the models, feature engineering was utilized on all three models. As much of the variance, and outliers, is due to most of the directors, writers, and stars only appearing once in the dataset, we set out to bin or group these names. It was found that grouping the directors, writers, and stars with one appearance into a separate category only made the model metrics worse. In the end, although we lost approximately 30% of the data, dropping these names improved each algorithm, but the random forest model produced the best results. Imputation was unfortunately not an option for these features as these features are difficult to estimate. Attempts to bin or group the other categorical variables, such as genre, MPAA rating, and company did not come to fruition, as the model metrics declined after tweaking these features. 

Following the binning and grouping of data, it became apparent that the linear regression model was not as accurate as the two others. Therefore, the linear regression model was dismissed at this point. Then, one-hot encoding, a form of feature engineering, was also utilized to convert the many categorical variables into a new binary variable to impute into the machine learning models. The data was then split into training and testing sets, and a standard scaler was used to scale the data.
To prevent overfitting, a grid search with cross validation was utilized to determine the best hyperparameters for the remaining two models. Several iterations of the grid search were performed, and it was ultimately found that the Random Forest model was the most robust out of the two, so we solidified it as the final model.
 
## Results
### Statistical Analysis
![boxplot-predicted](https://github.com/AleksKostrycka/Movie_Review_Project/blob/4fee0e4afcf183af4230741015e85ceef49ed1e0/Images/Boxplot%20of%20Predicted%20Scores.png)
### Machine Learning Model
The preliminary r² score for the linear regression, random forest, and gradient boost models were -2.55, 0.41, and 0.37, respectively. Further feature engineering of the linear regression machine learning model did not improve the metrics. The best metrics for the linear regression model was an r² score of 0.35, which was lower than the preliminary models of the random forest and gradient boost regression models, so the linear regression model was ultimately discarded.
After feature engineering of the remaining models as described above, the model metrics r², root mean square error (RMSE) and mean absolute error (MAE) came out to be 0.43, 0.81, and 0.47 for the random forest model and 0.37, 0.88, and 0.59 for the gradient boost model. After the grid search with cross validation, the random forest was determined to be the most robust supervised machine learning model for our prediction with an r² score of 0.43, RMSE of 0.76, and MAE of 0.45. This means that with our random forest supervised machine learning model, 43% of the variance can be explained by the original model, there is an average squared error of 0.76, and an average of 0.45 difference between the expected and actual IMDb ratings.

Although the r² score appears to be on the low side, it is actually expected with our dataset. Due to the arduous nature of the data and our analysis, we did not expect to find a great linear relationship between the features and the IMDb rating. Additionally, the other model metrics, RMSE and MAE, which measure the difference between the actual and predicted ratings, were fairly low, indicating an fairly accurate model. Furthermore, overfitting the data was not a large concern as we took steps to prevent this. 

![graph](https://github.com/AleksKostrycka/Movie_Review_Project/blob/4fee0e4afcf183af4230741015e85ceef49ed1e0/Images/Actual%20vs.%20Predicted%20Graph.png)
![chart](https://github.com/AleksKostrycka/Movie_Review_Project/blob/4fee0e4afcf183af4230741015e85ceef49ed1e0/Images/Actual%20vs.%20Predicted%20Ratings.png)

## Conclusion
In the end, our machine learning model was able to predict IMDb ratings with an average error of ±0.45. To reduce this error, more data points and features would be necessary. For future analysis, a Co-star feature would be recommended as many movies have more than one starring actor or actress. This will reduce the number of names that appears only once in the dataset, which can lead to a more robust prediction of IMDb rating. Additionally, it would have been preferable to have included the Budget feature into the machine learning model. Although there is a weak correlation between budget and the outcome of a movie, it would have been advantageous to include in the machine learning algorithm.

