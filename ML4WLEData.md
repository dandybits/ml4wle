# Prediction Assignment Writeup for JHU/Coursera Practical Machine Learning Class
dandybits  
**Synopsis:** This project involves analyzing of the Weight Lifting Exercises (WLE) Dataset. This document describes the approach for building classification model that allows to distinguish properly conducted weight lifting excercise movements from those conducted with common mistakes. 

This research was conducted as a test assignment for the [Data Science Certification on Coursera.](https://www.coursera.org/specializations/jhudatascienc)  

For more information about collection and original analyis of the WLE dataset see research article [Qualitative Activity Recognition of Weight Lifting Exercises]( http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz3ylUwfOl8) by Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. 

#### Exploring the WLE dataset
The first step in data analysis is loading the data and converting it into a data.table format that makes available a richer data processing capabilities than default dataframe format. 


```r
## load libraries
library(lattice)
library(ggplot2)
library(caret)

## load data into data frame
wle.data <- read.csv("data\\pml-training.csv")
```

#### Observation notes on WLE dataset
As described in the paper referenced above as well as revealed in exploring the data, the dataset contains records with various levels of granularity. There are 'timestamp'-level records that contain a set of sensor measurements as well as 'summary' records that contained averaged measurements for time windows of several sizes from 0.5 sec to 2.5 sec.
This makes the task somewhat ambiguous. We are trying to predict if a record belongs to a properly exectuted movement while any meaningful classification only applies at the level of the entire set of records for a particular movement. 

Moreover, since the surrogate identifier for the movement, num_window attribute, is present in the test data set, it is possible to predict based on the num_window attribute alone. 

While this may seem trivial, similar 'over_inclusive' datasets occasionally caused unintended results even in high-profile ML competitions.  

#### Predicting based on window_num only

```r
## splitting data for model validation
inTrain <- createDataPartition(wle.data$classe, p = 0.7, list = FALSE)
wle.train <- wle.data[inTrain,]
wle.test <- wle.data[inTrain,]
```

#### Predicting based on meaningful predictors 

#### Conclusions
