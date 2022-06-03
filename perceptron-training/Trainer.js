/*
Gradient decsent is a popular algorithm for solving AI problems
The goal of a linear regression is to fit a linear graph to a set of (X,Y) points

Start with a scatterplot and a linear model (y=wx+b)
Trains the model to find a line that fits the plot. This is done by altering the weight (slope) and bias(intercept) of the line

Create a trainer object that takes any number of (x,y) values in 2 arrays.
Set weight and bias to 0
A learning constant has to be set, and a cost variable defined.
*/


function Trainer(xArray, yArray) {
    this.xArr = xArray;
    this.yArr = yArray;
    this.points = this.xArr.length;
    this.learnc = 0.00001;
    this.weight = 0;
    this.bias = 0;
    this.cost;

    /*
Cost function that measures how good a solution is
Uses weight and bias and returns an error based on how well the line fits a plot
To compute the error, loop through all points in the plot and sum the square distances between the y value of each point and the line
The most conventional way is to square the distances (to ensure positive values) and to make error function differential.
*/

    this.costError = function () {
        let total = 0
        for (let i = 0; i < this.points; i++) {
            const y = this.yArr[i]
            const x = this.xArr[i]
            total += (y - (this.weight * x + this.bias)) ** 2
            return total / this.points
        }
    }

    /*
The gradient descent algorithm should walk the cost function towards the best line,
Each iteration should update both m and b towards a line with a lower cost (error)
To do that, we add a train function that loops over all the data many times
*/

    this.train = function (iter) {
        for (let i = 0; i < iter; i++) {
            this.updateWeights()
        }
        this.cost = this.costError();
    }

    /*
The train function should update the weights and biases in each iteration
The direction to move is calculated using two partial derivatives
*/

    this.updateWeights = function () {
        let w_deriv = 0
        let b_deriv = 0
        for (let i = 0; i < this.points; i++) {
            const y = this.yArr[i]
            const x = this.xArr[i]
            const wx = y - (this.weight * x + this.bias)
            w_deriv += -2 * wx * x
            b_deriv += -2 * wx
        }
        this.weight -= (w_deriv / this.points) * this.learnc
        this.bias -= (b_deriv / this.points) * this.learnc
    }

}






