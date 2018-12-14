rm(list = ls(all=T))
gc()
# library(RODBC)
# library(reshape2)
# library(dplyr)
# library(ggplot2)
# library(RMySQL)
# library(stringr)
# library(e1071) 
# library(tcltk)
# library(lubridate)
library(keras)
library(tensorflow)
library(corrplot)
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE) 
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
plot(iris$Petal.Length, 
     iris$Petal.Width, 
     pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], 
     xlab="Petal Length", 
     ylab="Petal Width")
M <- cor(iris[,1:4])

# Plot the correlation plot with `M`
corrplot(M, method="circle")

# Build your own `normalize()` function
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

# Normalize the `iris` data
iris_norm <- as.data.frame(lapply(iris[1:4], normalize))

iris[,5] <- as.numeric(iris[,5]) -1

# Turn `iris` into a matrix
iris <- as.matrix(iris)

# Set `iris` `dimnames` to `NULL`
dimnames(iris) <- NULL
iris[,1:4] <- normalize(iris[,1:4])
summary(iris)

# Determine sample size
set.seed(1)
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))

# Split the `iris` data
iris.training <- iris[ind==1, 1:4]
iris.test <- iris[ind==2, 1:4]
# Split the class attribute
iris.trainingtarget <- iris[ind==1, 5]
iris.testtarget <- iris[ind==2, 5]

# One hot encode training target values
iris.trainLabels <- to_categorical(iris.trainingtarget)
# One hot encode test target values
iris.testLabels <- to_categorical(iris.testtarget)

model<-keras_model_sequential()
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')


# Print a summary of a model
summary(model)
get_config(model)
get_layer(model, index = 1)
model$layers
model$inputs
model$outputs


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 200,
  batch_size = 5, 
  validation_split = 0.2
)

# Plot the history
plot(history)


plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")

# Plot the model loss of the test data
lines(history$metrics$val_loss, col="green")

# Add legend
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))


plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")

# Plot the accuracy of the validation data
lines(history$metrics$val_acc, col="green")

# Add Legend
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

classes <- model %>% predict_classes(iris.test, batch_size = 128)
table(iris.testtarget, classes)



# Evaluate on test data and labels
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
# Print the score
print(score)

# Initialize the sequential model
model <- keras_model_sequential() 
# Add layers to model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model to the data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)



model <- keras_model_sequential() 
# Add layers to the model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Save the training history in history
history <- model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5,
  validation_split = 0.2
)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))





model <- keras_model_sequential() 
# Add layers to the model
model %>% 
  layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')
# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)
# Fit the model to the data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)



model <- keras_model_sequential() 
# Add layers to the model
model %>% 
  layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Save the training history in the history variable
history <- model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))



# Initialize a sequential model
model <- keras_model_sequential() 
# Build up your model by adding layers to it
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')
# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)

# Use the optimizer to compile the model
model %>% compile(optimizer=sgd, 
                  loss='categorical_crossentropy', 
                  metrics='accuracy')

# Fit the model to the training data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)
# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
# Print the loss and accuracy metrics
print(score)


sgd <- optimizer_sgd(lr = 0.01)
# Compile the model
model %>% compile(optimizer=sgd, 
                  loss='categorical_crossentropy', 
                  metrics='accuracy')

# Fit the model to the training data
history <- model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))


save_model_hdf5(model, "my_model.h5")
model <- load_model_hdf5("my_model.h5")
save_model_weights_hdf5("my_model_weights.h5")
model %>% load_model_weights_hdf5("my_model_weights.h5")

json_string <- model_to_json(model)
model <- model_from_json(json_string)
yaml_string <- model_to_yaml(model)
model <- model_from_yaml(yaml_string)