import numpy as np
class sigmoidperceptron():
    def __init__(self,input_size):
        self.weights=np.random.randn(input_size)
        self.bias=np.random.randn(1)




    def sigmoidz(self,z):
        return 1/(1+np.exp(-z))


    def predict(self,inputs):
        weighted_sum=np.dot(inputs,self.weights)+self.bias
        return self.sigmoidz(weighted_sum)


    def fit(self,inputs,targets,learning_rate,num_epochs):

        num_examples=inputs.shape[0]
        for epoch in range(num_epochs):
            for i in range(num_examples):
                inputvektor=inputs[i]
                target=targets[i]
                prediction=self.predict(inputvektor)
                error=target-prediction
                gradientwei=error*prediction*(1-prediction)*inputvektor
                self.weights+=learning_rate*gradientwei
                gradientbias=error*prediction*(1-prediction)
                self.bias+=learning_rate*gradientbias




    def evaluate(self,inputs,targets):
        correct=0
        for inputvektor,target in zip(inputs,targets):
            predictin=self.predict(inputvektor)
            if predictin>=0.5:
                predictclas=1

            else:
                predictclas=0

            if predictclas==target:
                correct+=1

            accuracy=correct/len(inputs)
            return accuracy

