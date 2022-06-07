/*
Experiment with: 
 - changing the number of epochs. How many do you need before the graph flattens out?
 - increasing the number of units in the hidden layer
 - adding more hidden layers in between the first and the output. 
*/

async function getData() {
    const carsDataResponse = await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json")
    const carsData = await carsDataResponse.json()
    const cleaned = carsData
        .map((car) => ({ mpg: car.Miles_per_Gallon, horsepower: car.Horsepower }))
        .filter((car) => car.mpg !== null && car.horsepower !== null)
    return cleaned
}

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

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 50, useBias: true }))
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 100, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 1, useBias: true }))
    return model
}

const model = createModel();
tfvis.show.modelSummary({ name: "Model Summary" }, model);

function convertToTensor(data) {

    return tf.tidy(() => {
        tf.util.shuffle(data)
        const inputs = data.map(d => d.horsepower)
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labels = data.map(d => d.mpg)
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()
        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()
        const normalisedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalisedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalisedInputs,
            labels: normalisedLabels,
            inputMax, inputMin, labelMax, labelMin
        }
    });
}

async function trainModel(model, inputs, labels) {
    model.compile({ optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError, metrics: ["mse"] });
    const batchSize = 32
    const epochs = 100

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


function testModel(model, inputData, normalisationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalisationData;
    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100)
        const preds = model.predict(xs.reshape([100, 1]))

        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

        return [unNormXs.dataSync(), unNormPreds.dataSync()]
    });

    const queryHorsepower = 120
    const queryPoint = getMPG(queryHorsepower, model, normalisationData)
    const predictedPoints = Array.from(xs).map((val, i) => { return { x: val, y: preds[i] } });
    const originalPoints = inputData.map(d => ({ x: d.horsepower, y: d.mpg }));

    tfvis.render.scatterplot(
        { name: "Model Predictions v Original Data" },
        { values: [originalPoints, predictedPoints, queryPoint], series: ["Original", "Predicted", "Query"] },
        { xLabel: "Horsepower", yLabel: "MPG", height: 300 }
    )
}

function getMPG(horsepower, model, normalisationData) {
    console.log(horsepower)
    const { inputMax, inputMin, labelMin, labelMax } = normalisationData;
    const [query, queryPred] = tf.tidy(() => {
        const query = [horsepower]
        const queryTensor = tf.tensor2d(query, [query.length, 1])
        const normQueryTensor = queryTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const queryPred = model.predict(normQueryTensor.reshape([1, 1]))

        const unNormQ = normQueryTensor.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormQP = queryPred.mul(labelMax.sub(labelMin)).add(labelMin);

        return [unNormQ.dataSync(), unNormQP.dataSync()]
    });
    console.log(`The MGP for a vehicle with ${horsepower} horsepower is ${queryPred} MPG`)
    return Array.from(query).map((val, i) => { return { x: val, y: queryPred[i] } })
}


/*
Increasing number of epochs:
 - increased to 150
 - decreased final loss to 0.016972 from 0.024791 at 50 epochs
 - on 50 epochs, we get massive variety in predictions, including some with a positive trend
 - with 150 epochs, there is a much more stable negative trend upon re-running the model

Increasing units in hidden layer from 1 to 10:
 - not much change on the prediciton trajectory, but much more stable than with 1 unit
 - More irregular drop in loss on epochs

Increasing the number of layers:
 - disrupts the smooth loss curve


 The above combination produces a prediction that follows the curve of the data as opposed to being a linear line



*/