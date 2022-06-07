
/*
Make a webpage that uses TensorFlow.js to train a model in the browser. Given Horsepower for a car, the model will learn to predict "miles per gallon"

1. Load the data and prepare it for training
2. Define the architecture of the model
3. Train the model and monitor its performance as it trains
4. Evaluate the trained model by making some predictions
*/
// console.log("Hello TensorFlow")

/*
Get the car data reduced to just the variables we are interested and cleaned of missing data
*/

async function getData() {
    const carsDataResponse = await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json")
    const carsData = await carsDataResponse.json()
    const cleaned = carsData
        .map((car) => ({ mpg: car.Miles_per_Gallon, horsepower: car.Horsepower }))
        .filter((car) => car.mpg !== null && car.horsepower !== null)
    return cleaned
}

/*
Let's plot this data in a scatterplot to see what it looks like
*/

async function run() {
    const data = await getData()
    const values = data.map((d) => ({ x: d.horsepower, y: d.mpg }))
    tfvis.render.scatterplot({ name: "Horsepower v MPG" }, { values }, { xLabel: "Horsepower", yLabel: "MPG", height: 300 });

    const tensorData = convertToTensor(data)
    const { inputs, labels } = tensorData

    await trainModel(model, inputs, labels);
    console.log("Training Complete")

    testModel(model, data, tensorData)
}

document.addEventListener('DOMContentLoaded', run);

/*
Negative correlation between HP and MPg -> as horsepower increases, generally cars have fewer miles per gallon
*/

/*
Our goals is to train a model that will take one number, HP, and learn to predict one number, MPG. This is one-to-one mapping
We will feed these examples to a neural network that will learn from these examples a formula to predict MPG given horsepower. This learning
from examples for which we have the correct answers is called supervised learning
*/


/*
Model architecture -> which functions will the model run when it is executing / what alorithm will our model use to compute its answers?
ML models are algorithms that take an input and produce an output. When using neural networks, the algorithm is a set of layers of neurons with "weights" 
governing their output. The training process learns the ideal values for those weights
*/

function createModel() {
    //create a sequential model
    const model = tf.sequential();
    //add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
    //add an output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }))
    return model
}

/*
const model = tf.sequential();
    This instantiates a tf.Model object. this model is sequential because its inputs flow straight down to its output.
    Other kinds of models can have branches, or even multiple inputs and outputs, but in many cases your models will be sequential

model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
    This adds an input layer to our network, which is automatically connected to a dense layer with one hidden unit. A dense layer is a type of layer that
    multiplies its inputs by a matrix (called weights) and then adds a number (bias) to the result. As this is the first layer of the network, we
    need to define our input shape. The input shape is [1] because we have 1 number as our input (the horsepower of a car)
    units sets how big the weight matric will be in the layer. By setting it to 1 here we are saying there will be 1 weight for each of the input features of
    the data.
    Dense layers come with a bias term by default, so we do not need to set useBias to true, we will omit from further calls

model.add(tf.layers.dense({ units: 1, useBias: true }))
    The code above creates our output layer. We set units to 1 because we want to output 1 number
    In this example, because the hidden layer has 1 unit, we don't actually need the final outpiut layer above (we could use the hidden layer as the output layer)
    However, defining a separarte output layer allows us to modify the number of units in the hidden layer wile keeping the one-to-one mapping of input and output
*/

//Create the model
const model = createModel();
tfvis.show.modelSummary({ name: "Model Summary" }, model);

/*
To get the performance benefits of TensorFlow.js that make training ML models practical, we need to convert our data to tensors. We will also perform a number of tranformations on 
our data that are best practices, namely shuffling and normalisation.
*/

/**
 * Convert the input data to tensors that we can use for ML. 
 * Also shuffle and normalise the data
 * MPG on y-axis
 * @param {*} data 
 */

function convertToTensor(data) {
    //wrapping these calculations in a tidy will dispose any intermdeiate tensors
    return tf.tidy(() => {
        //Step 1: Shuffle the data
        tf.util.shuffle(data)

        //Step 2: Convert data to Tensor
        const inputs = data.map(d => d.horsepower)
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labels = data.map(d => d.mpg)
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3: Normalize the data to the range 0-1 using min-max scaling
        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()
        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()

        const normalisedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalisedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        //Step 4: return all the necessary information
        return {
            inputs: normalisedInputs,
            labels: normalisedLabels,
            inputMax, inputMin, labelMax, labelMin
        }
    });
}

/*
tf.util.shuffle(data)
    randomise the order of the sxamples we will feed to the training algorithm. Shuffling is important because typically during training
    the dataset is broken up into smaller subsets, called batches, that the model is trained on. Shuffling helps each batch have a variety of data 
    from accross the data distribution. By doing so we help the model:
     - not learn things that are purely dependent on the order the data was fed in
     - not be sensitive to the structure in subgroups(eg if it only see high horsepower cars for the first half of its training it may learn a 
        relationship that does not apply across the rest of the dataset)

const inputs = data.map(d => d.horsepower)
const inputTensor = tf.tensor2d(inputs, [inputs.length], 1);
const labels = data.map(d => d.mpg)
const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    Here we make two arrays, one for our input examples (HP) and another for the true output values (labels => MPG)
    We then convert each array data to a 2d tensor. The tensor will have a shape of [num_examples, num_features_per_Example]. Here we have inputs.length
    examples and each example has 1 input feature (HP)

const inputMax = inputTensor.max();
const inputMin = inputTensor.min();
const labelMax = labelTensor.max();
const labelMin = labelTensor.min();
const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
    Next, we normalise the data. It's normalised into the numerical range 0-1 using min-max scaling. Normalisation is important because the internals of many
    machine learning models are designed to work with numbers that are not too big. Common ranges to normalise data to include 0-1 or -1 to 1. You will have more 
    success training your models if you get into the habit of normalizing your data to some reasonable range.
    Some datasets will be learned without normalisation, but doing so will eliminate a whole class of problems that would prevent effective learning.
    You can normalise your data before turning it into tesnors. We do it afterwards because we can take advantage of vectorisation in TensforFlow.js to do the 
    min-max scaling operations without writing any explicit for-loops

    We want to keep the values we used for normalisation during training so that we can un-normalise the outputs get them back into our original scale
    and to allow us to normalise future input data in the same way
*/


/*
With our model instance created and our data represented as tensors we have everything in place to start the training process
*/

async function trainModel(model, inputs, labels) {
    //Prepare model for training
    model.compile({ optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError, metrics: ["mse"] });
    const batchSize = 32
    const epochs = 50

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: "Training Performance" },
            ["loss", "mse"],
            { height: 200, callbacks: ["onEpochEnd"] }
        )
    })
}

/*
We have to compile the model before we train it. To do so, we have to specify a number of very important things:
 - Optimizer :
    this is the algorithm that is going to govern the updates to the model as it sees examples. There are many optimizers available in TensorFlow.js
    here we picked the adam optimizer as it is quite effective in practice and requires no configuration
 - Loss :
    this is a function that will tell the model how well it is doing on learning each of the batches that it is shown. Here we use meanSquareError to
    compare the predictions made by the model with the true values

const batchSize = 32;
const epochs = 50;
    Next, we pick a batchSize and a number of epochs
    - batchSize :
        refers to the size of the data subsets that the model will see on each iteration of training. Common batch sizes tend to be in the range 32-512. There
        isn't really an ideal batch size for all problems
    - epochs :
        refers to the number of times the model is going to look at the entire dataset that you provide it

model.fit is the function we call to start the training loop. It is an asynchronous function so we return the promise it gives us so that the caller can determine when training is complete
To monitor training progress we pass some callback to model.fit. We use tfvis.show.fitCallbacks to generate functions that plot charts for the loss and mse metric we specify above

*/

//Now call the function we have defined from our run function
//Lines 34-38 added to run function


/*
The two graphs are created by the callbacks. They display the loss and mse, averaged over the whole dataset,at the end of each epoch
When training a model we want to see the loss go down. In this case, because our metric is a measure of error, we want to see if decrease too
*/


/*
Now that our model is trained, we want to make some predictions. Let's evaluate the model by seeing what it predicts for a uniform range of numbers
from low to high horsepowers
*/

function testModel(model, inputData, normalisationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalisationData;

    //Generate predictions for a uniform range of numbers between 0 and 1
    //We un-normalise the data by doing the inverse of the min-max scaling that we did earlier
    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100)
        const preds = model.predict(xs.reshape([100, 1]))

        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

        return [unNormXs.dataSync(), unNormPreds.dataSync()]
    });

    const predictedPoints = Array.from(xs).map((val, i) => { return { x: val, y: preds[i] } });
    const originalPoints = inputData.map(d => ({ x: d.horsepower, y: d.mpg }));

    tfvis.render.scatterplot(
        { name: "Model Predictions v Original Data" },
        { values: [originalPoints, predictedPoints], series: ["Original", "Predicted"] },
        { xLabel: "Horsepower", yLabel: "MPG", height: 300 }
    )
}

/*
const xs = tf.linspace(0, 1, 100);
const preds = model.predict(xs.reshape([100, 1]));
    We generate 100 new examples to feed to the model. Model.predict is how we feed those examples into the model. Note that ehy need to have a similar
    shape [num_examples, num_features_per_example] as when we did training

const unNormXs = xs
.mul(inputMax.sub(inputMin))
.add(inputMin);

const unNormPreds = preds
.mul(labelMax.sub(labelMin))
.add(labelMin);
    To get the data back to our original range (rather than 0-1) we use the values we calculated while normalising, but just invert the operations

.dataSync()
    method we can use to get a typed array of the values stored in a tensor. This allows us to process those values in regular javaScript
    This is a synchronous version of the .data() method which is generally preferred

Finally, we use tfjs-vis to plot the original data and the predictions from the model
*/

//Make some predictions using the model and compare them to the original data
//Add line 40 to run function