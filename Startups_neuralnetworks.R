install.packages("neuralnet")
install.packages("nnet")
library(readr)
library(neuralnet)
library(nnet)
library(DataExplorer)
library(plyr)
library(ggplot2)
library(psych)

#Importing Dataset
Startups <- read.csv("C:\\Users\\91755\\Desktop\\Assignment\\11 - Neural Network\\50_Startups.csv")
attach(Startups)
head(Startups)

#EDA and Statistical Analysis
sum(is.na(Startups))
str(Startups)
class(Startups)
table(Startups$State)
Startups$State <- as.numeric(revalue(Startups$State, c("California"="0", "Florida"="1", "New York"="2")))
summary(Startups)

#Graphical Representation
table(Startups)
pairs(Startups)
plot(State, Profit)
plot(Administration, Profit)
pairs.panels(Startups)
ggplot(Startups, aes(x=R.D.Spend, y=Profit))+geom_point()

#Normalization
normal <- function(x)
{
  return((x-min(x))/(max(x)-min(x)))
}
Startups_nor <- as.data.frame(lapply(Startups, FUN=normal))
head(Startups_nor)
summary(Startups$Profit)

#Data Splitting
set.seed(123)
split <- sample(2, nrow(Startups_nor), replace = T, prob = c(0.75, 0.25))
Startnorm_train <- Startups_nor[split==1,]
Startnorm_test <- Startups_nor[split==2,]
head(Startnorm_train)

#Model Building
set.seed(333)
Model_1 <- neuralnet(Profit~., data = Startnorm_train)
summary(Model_1)
plot(Model_1, rep = "best")

#Evaluation
set.seed(123)
Model1_result <- compute(Model_1, Startnorm_test[,1:4])
Model1_result
pred_1 <- Model1_result$net.result
cor(pred_1, Startnorm_test$Profit)          #Accuracy = 96.02%

#Since the prediction on profit is in the normalised form.
#To compare, need to denormalise the predicted profit value
startup_min <- min(Startups$Profit)
startup_max <- max(Startups$Profit)
denormalize <- function(x, min, max){
  return(x*(max-min)+min)
}
Profit_pred <- denormalize(pred_1, startup_min, startup_max)
data.frame(head(Profit_pred), head(Startups$Profit))

#Model Building Using Two Hidden Layers
set.seed(1234)
Model_2 <- neuralnet(Profit~., data = Startnorm_train, hidden = 2)
str(Model_2)
plot(Model_2, rep = "best")

#Evaluation
set.seed(333)
Model2_result <- compute(Model_2, Startnorm_test[,1:4])
Model2_result
pred_2 <- Model2_result$net.result
cor(pred_2, Startnorm_test$Profit)    #Accuracy = 96.88%

#Model Buiding Using Five Hidden Layers
set.seed(2222)
Model_3 <- neuralnet(Profit~., data = Startnorm_train, hidden = 6)
str(Model_3)
plot(Model_3, rep = "best")

#Evaluation
set.seed(4444)
Model3_result <- compute(Model_3, Startnorm_test[1:4])
Model3_result
pred_3 <- Model3_result$net.result
cor(pred_3, Startnorm_test$Profit)      #Accuarcy=96.26%
#Accuracy is increase by increasing the hidden layers