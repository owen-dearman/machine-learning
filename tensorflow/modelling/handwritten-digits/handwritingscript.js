/*
RUN WITH LIVE SERVER
*/

import { MnistData } from './data.js';

/*
There are a total of 65,000 images. We will use 55,000 of them to train the model, saving 10,000 images that we can
use to test the model's performance once we are done.
*/

async function showExamples(data) {
    // Create a container in the visor
    const surface =
        tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' });

    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

async function run() {
    const data = new MnistData();
    await data.load();
    await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);

    await train(model, data);
    await showAccuracy(model, data);
    await showConfusion(model, data);
}

document.addEventListener('DOMContentLoaded', run);

/*
Our goal is to train a model that will take one image and learn to predict a score for each of the 
possible 10 classes that image may belong to (digits 0-9)
Each image is 28px wide and 29px high and has 1 colour channel. The shape of each image is
[28,28,1]
We do a 1-to-10 mapping, as well as the shape of each input example
*/

/*
In this section we will write code to describe the model architecture. We define an architecture and let
the training process learn the parameters of that algorithm

*/

function getModel() {
    const model = tf.sequential();
    const imageWidth = 28
    const imageHeight = 28
    const imageChannels = 1

    //in the first layer of our convolutional neural network we have to specify the input
    //shape. Then we specify some parameters for the convolution operation that takes place
    //in this layer 

    model.add(tf.layers.conv2d({
        inputShape: [imageWidth, imageHeight, imageChannels],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
    }));

    //The MaxPooling layer acts as a downsampling using max values in a region instead of averaging
    model.add(tf.layers.maxPooling2d({ poolSize: (2, 2), strides: (2, 2) }));

    //repeat another conv2d + maxPooling stack. Nb. more filters in the convolution
    model.add(tf.layers.conv2d({
        inputShape: [imageWidth, imageHeight, imageChannels],
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: (2, 2), strides: (2, 2) }));

    //now we flatten the output from the 2d filterss into a 1d vector to prepare it for input
    // into our last layer. This is common practice when feeding higher dimensional data to a final
    //classification output layer
    model.add(tf.layers.flatten())

    //our last layer is a dense layer which has 10 output units, one for each output class
    const numOutputClasses = 10
    model.add(tf.layers.dense({
        units: numOutputClasses,
        kernelInitializer: "varianceScaling",
        activation: "softmax"
    }));

    //choose an optimizer, loss function and accuracy metric.
    //compile and return the model
    const optimizer = tf.train.adam()
    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });

    return model
}

/*
We are using a sequential model
We are using a conv2d layer instead of a dense layer.

inputShape
    shape of the data that will flow into the first layer of the model. Our MNIST examples are 28x28 B&W images
    The canonical format for image data is [row, column, depth]. We do not specify a batch size in the input shape. Layers 
    are designed to be batch size agnostic so that during inference you can pass a tensor of any batch size in.
kernelSize
    The size of the sliding convolutional filter windows to be applied to the input data. Here, we set a kernel size of 5
    which specifies a square 5x5 convolutional window
filters
    the number of filter windows of size kernelSize to apply to the input data. Here we will apply 8 filters to the data
strides
    The "step-size" of the sliding window - how many pixels the filter will shift each time it moves over the image. Here we 
    specify strides of 1, which means that the filter will slide over the image in steps of 1 pixel
activation
    the activation function to apply to the data after the convolution is complete. In this case, we are applying a 
    rectified linear unit (ReLU) function which is a very common activation function in ML models
kernelInitializer
    the method uses for randomly initializing the model weights which is very important to training dynamics. Variance Scaling is generally 
    a good choice
    
You can build an image classifier using only dense layers, however, convolutional layers have proven effective for many image based tasks
Images are high dimensional data, and convolution operations tend to increase the size of the data that went into them. Before passing them to 
our final classification layer we need to flatten the data into one long array.
Dense layers only take tensor1ds so this step is common in many classification tasks
There are no weights in a flatten layer

We use a dense layer with a softmax activation to compute probability distributions over the 10 possible classes
The class with the highest score will be the predicted digit.
Softmax is the most likely activation you will want to use at the last layer of a classification task
We want 10 units in our output layer, because we are doing a 1-to-10 mapping

We compile the model using optimizer, loss and metrics we want to keep track of
In contrast to MPG tutorial, we use categoricalCrossentropy as our loss function. This is used when the output of the model
is a probability distribution generated by the last layer of our model and the probability distribution given by our last true label.
Categorical cross entropy will produce a single number indicating how similar the prediction vector is to our true label vector
The data representation used here for the labels is called one-hot encoding and is common in classification problems. Each class has a probability
associated with it for each example. When we know exactly what it should be we can set that probability to 1 and the others to 0

The other metric we monitor is accuracy which for a classification problem is the percentage of correct predictions out of all predictions

*/

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const batchSize = 512;
    const trainedDataSize = 6000;
    const testedDataSize = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(trainedDataSize);
        return [
            d.xs.reshape([trainedDataSize, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(testedDataSize);
        return [
            d.xs.reshape([testedDataSize, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: batchSize,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

/*
Metrics :
    we decide which metrics we are going to monitor. We will monitor loss and accuracy
    on the training set as well as loss and accuracy on the validation set
    When using layers API loss and accuracy is computed on each batch and epoch

Prepare Data as tensors:
    we make 2 datasets, a training set that we will train the model on,  and a validation set
    that we will test the model on at the end of each epoch. However, the data in the validation set 
    is never shown to the model during training

    The data class we provided makes it easy to get tensors from the image data. But we will still reshape
    the tensors into the shape expected by the model before we can feed these to the model. For each dataset, we have
    both inputs and labels

Return
    We call model.fit to start the training loop. We pass a validationData proerty to indicate which data the model should
    use to test itself after each epoch (but not use for training)
    If we do well on our training data but not on our validation data, it means the model is likely overfitting to the training
    data and won't generalise well to input it has not previously seen

*/


/*
The validation accuracy provides a good estimate on how well our model will do 
on data it hasn't seen before (as long as that data resembles the validation set in some way). 
However we may want a more detailed breakdown of performance across the different classes.

There are a couple of methods in tfjs-vis that can help you with this.

*/
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
    const imageWidth = 28;
    const imageHeight = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, imageWidth, imageHeight, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels];
}

/*
What is this code doing?

Makes a prediction.
Computes accuracy metrics.
Shows the metrics
Let's take a closer look at each step.
*/


async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    labels.dispose();
}

async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });

    labels.dispose();
}

/*
First we need to make some predictions. Here we will make take 500 images and predict what
digit is in them (you can increase this number later to test on a larger set of images).

Notably the argmax function is what gives us the index of the highest probability class. 
Remember that the model outputs a probability for each class. Here we find out the highest 
probability and assign use that as the prediction.

You may also notice that we can do predictions on all 500 examples at once. This is the 
power of vectorization that TensorFlow.js provides.


*/