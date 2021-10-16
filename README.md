# Support Vector Machine
Support Vector Machine is a great tool either for classification or for regression problems since it allows to work efficiently with outliers and is a great tool to compromise the bias-variance trade-off. It uses the kernel function with D=dimension to transform the model from low (1 dimensional) to high(2–3) dimensional observations or radial kernel (Similar to KNN takes the nearest observation) for the more advanced models.
## Support Vector Classifiers
Imagine you are an HR manager in a Security company and you need to recruit guys for the bodyguard positions. The main criterion is height. So you classify the observations as X≤1.80cm ≤ Y. In this one-dimensional dataset, the threshold is so simple to identify. Indeed, you don’t need even to use Support Vector Classifier. <br />
![image](https://user-images.githubusercontent.com/90762709/137595802-b2fb447f-730a-4044-8aa4-bca1def2fe67.png)
<br />
But what if you have a complex dataset with various variables?Imagine below we are trying to find whether the person has diabetes or not. In this case, the Support Vector Classifiers can help you with the task. In the picture below you see again the classification model however, this time classification is done through Support Vector Classifier and the threshold is established by the maximal distance between two classes of the observation Called the Maximal Margin Classifier. <br />
![image](https://user-images.githubusercontent.com/90762709/137595826-bc507e0d-708d-4027-a14f-7dd6dbb6ff60.png)
<br />
But what if one observation with diabetes will occur among those with not diabetes as in the picture below?
In this case, the SVC can tolerate the misclassification for the sake of the Bias-Variance Trade-Off. Meaning the classifier will tolerate the diabetes data among not diabetes and will classify it as it doesn’t exist. <br />
![image](https://user-images.githubusercontent.com/90762709/137595836-2cc5b7fb-591b-4b72-9b36-25f26d0ec0a1.png)
<br />
## Support Vector Machines
Although the above-mentioned technique is great to classify two-dimensional data, sometimes we have a dataset that is not subject to the classification with a simple threshold. In this case, it’s time to use Support Vector Machines.
We will use the built-in Iris dataset in R and visualize it in order to see how the observations are located on the dataset. This dataset will work greatly for SVM since it has observations that are overlapping. So the quadratic or cubic polynomial model would work with it greatly. <br />
```
data = iris
pairs(data[,1:4], pch = 19,  cex = 0.5,
      col = data$Species)
```
![image](https://user-images.githubusercontent.com/90762709/137595882-7e60cf5f-0fe0-47cc-b7ee-889812033dda.png)
<br />
Now after visualizing the dataset we will try to run a polynomial kernel function with a 10-fold Sampling technique and will choose the best model.
Let’s see some examples of how the polynomial function will work with the dataset where the classification is not straightforward and the are lots of overlaps.<br />
![image](https://user-images.githubusercontent.com/90762709/137595901-e27bdcab-f8dc-4a79-995a-8d21e68910ae.png) <br />

Coming back to our Iris dataset. These are the libraries you will need to download. <br />
```
library(tidyverse)
library(dplyr)
library(ggplot2)
library(e1071)
```
Once you have uploaded them. We need to split our dataset into training and test subsets through which we will train and predict and eventually find the accuracy of our model. 

```
#Train-Test data----
set.seed(1,sample.kind = "Rejection")
tr_index = sample(1:nrow(data), nrow(data)*0.7)
train_data = data[tr_index,]
test_data = data[-tr_index,]
```
<br />
I love this package because it has a built-in 10-fold cross-validation function which helps you to get the best parameters of cost and degree. <br />
```
set.seed(1,sample.kind = "Rejection")
tune.out=tune(svm, #Tuning
              Species ~ .,
              data = train_data,
              kernel="polynomial",
              ranges=list(cost=c(0.1,1,10,100,1000),
                          degree=1:5))
summary(tune.out)
svm.pred.best = predict(tune.out$best.model,
                        newdata=test_data)
table(svm.pred.best, test_data$Species)
```
Accuracy of the model. Whenever the value is true it returns true. True is calculated as 1 and for mean you will get 1 if the number of the predictions are equal to the number of the actual test dataset response variables. 

```
mean(svm.pred.best == test_data$Species)

```
For radial kernel, you just need to write ‘radial’ instead of ‘polynomial’ 


```
svm1 = svm(Species~.,
           data = train_data,
           kernel = "radial",
           cost = 1,
           gamma = 1)
```

