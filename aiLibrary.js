function Perceptron(n, learningRate) {
    this.learnc = learningRate
    this.bias = 1

    this.weights = []
    for (let i = 0; i <= n; i++) {
        this.weights[i] = Math.random() * 2 - 1
    }

    this.activate = function (inputs) {
        let sum = 0
        for (let i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights[i]
        }
        return sum > 0 ? 1 : 0
    }

    this.train = function (inputs, desired) {
        inputs.push(this.bias);
        let guess = this.activate(inputs)
        let error = desired - guess
        if (error !== 0) {
            for (let i = 0; i < inputs.length; i++) {
                this.weights[i] += this.learnc * error * inputs[i]
            }
        }

    }
}


function Trainer(xArray, yArray) {
    this.xArr = xArray;
    this.yArr = yArray;
    this.points = this.xArr.length;
    this.learnc = 0.00001;
    this.weight = 0;
    this.bias = 0;
    this.cost;

    this.costError = function () {
        let total = 0
        for (let i = 0; i < this.points; i++) {
            const y = this.yArr[i]
            const x = this.xArr[i]
            total += (y - (this.weight * x + this.bias)) ** 2
            return total / this.points
        }
    }

    this.train = function (iter) {
        for (let i = 0; i < iter; i++) {
            this.updateWeights()
        }
        this.cost = this.costError();
    }

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