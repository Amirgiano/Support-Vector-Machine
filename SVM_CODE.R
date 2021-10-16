library(tidyverse)
library(dplyr)
library(ggplot2)
library(e1071)


data = iris

head(iris)


pairs(data[,1:4], pch = 19,  cex = 0.5,
      col = data$Species)

#Train-Test data----
set.seed(1,sample.kind = "Rejection")
tr_index = sample(1:nrow(data), nrow(data)*0.7)
train_data = data[tr_index,]
test_data = data[-tr_index,]

#Model----
#Train
svm.fit = svm(Species ~ .,
              data = train_data,
              kernel = "polynomial",
              cost = 1,
              degree = 1)
#Test
svm.pred = predict(svm.fit,
                   test_data)

#Validation
table(svm.pred, test_data$Species)
mean(svm.pred==test_data$Species)

#0.96 is a great accuracy result.However we will 

#Fortunately on SVM we can adjust the size of the cost and degree without building a function
#Technically it's called #sampling method: 10-fold cross validation
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

mean(svm.pred.best == test_data$Species)

#Now Radial Kernel
svm1 = svm(Species~.,
           data = train_data,
           kernel = "radial",
           cost = 1,
           gamma = 1)
svm1

#predictions
pred_svm1 = predict(svm1, newdata=test_data)
#svm1 is a vector
#Accuracy
mean(pred_svm1 == test_data$Species)


