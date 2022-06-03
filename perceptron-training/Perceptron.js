/*
Perceptron starts with a random weight for each input
For each mistake while training the perceptron, the weights will be adjusted with a small fraction
This is called the learning rate (learnc)
Sometimes, if both inputs are 0, the perceptron may produce an incorrect result
We give it a bias to avoid this
*/

function Perceptron(n, learningRate) {
    this.learnc = learningRate
    this.bias = 1

    //create random weights for each input
    this.weights = []
    for (let i = 0; i <= n; i++) {
        this.weights[i] = Math.random() * 2 - 1
    }

    /*
    add an activate function
    Multiply each input with the perceptron's weights
    sum the results
    compute the outcome
    */
    this.activate = function (inputs) {
        let sum = 0
        for (let i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights[i]
        }
        return sum > 0 ? 1 : 0
    }

    /*
    the training function guesses the outcome based on the activate function
    everytime the guess is wrong, the perceptron should adjust the weights
    after many guesses and adjustments, the weights will be correct
    */

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

/*
After each guess, the perceptron calculates how wrong the guess was
If it is wrong, the perceptron adjusts the bias and weights so that it will be more correct next time
This is called backpropagation
It will become a good guesser after a few thousand guesses

*/