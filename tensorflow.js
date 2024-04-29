/** This function plots the square footage and prices
 * 
 * @param {number[]} pointsArray 
 * @param {string} featureName 
 * @param {number[]} predictedPointsArray 
 */

async function plot(pointsArray, featureName, predictedPointsArray = null) {
    const values = [pointsArray.slice(0, 1000)];
    const series = ["original"];
    if (Array.isArray(predictedPointsArray)) {
        values.push(predictedPointsArray);
        series.push("predicted");
    }

    tfvis.render.scatterplot(
        { name: `${featureName} vs House Price` },
        { values, series },
        {
            xLabel: featureName,
            yLabel: "Price",
            height: 300,
        }
    )
}

/** This function plots the predicted curve on the graph
 * 
 */
async function plotPredictionLine() {
    const [xs, ys] = tf.tidy(() => {
        const normalizedXs = tf.linspace(0, 1, 100);
        const normalizedYs = model.predict(normalizedXs.reshape([100, 1]));

        const xs = denormalize(normalizedXs, normalizedFeature.min, normalizedFeature.max);
        const ys = denormalize(normalizedYs, normalizedLabel.min, normalizedLabel.max);

        return [xs.dataSync(), ys.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, index) => {
        return { x: val, y: ys[index] };
    });

    await plot(points, "Square feet", predictedPoints);
}


/**
 * Normalizes a tensor by subtracting the minimum value and dividing by the range.
 * If previousMin and previousMax are provided, they will be used instead of calculating the min and max from the tensor.
 * @param {tf.Tensor} tensor - The input tensor to be normalized.
 * @param {tf.Tensor} previousMin - The previous minimum value. (Optional)
 * @param {tf.Tensor} previousMax - The previous maximum value. (Optional)
 * @returns {Object} - An object containing the normalized tensor, minimum value, and maximum value.
 */
function normalize(tensor, previousMin = null, previousMax = null) {
    const min = previousMin || tensor.min();
    const max = previousMax || tensor.max();
    const normalizedTensor = tensor.sub(min).div(max.sub(min));
    return {
        tensor: normalizedTensor,
        min,
        max
    };
}

/**
 * Denormalizes a tensor using the given min and max values.
 * 
 * @param {tf.Tensor} tensor - The tensor to denormalize.
 * @param {number} min - The minimum value used for normalization.
 * @param {number} max - The maximum value used for normalization.
 * @returns {tf.Tensor} The denormalized tensor.
 */
function denormalize(tensor, min, max) {
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
    return denormalizedTensor;
}

let model;

/**
 * Creates a TensorFlow.js model for linear regression.
 * @returns {tf.Sequential} The created model.
 */
function createModel() {
    model = tf.sequential();

    model.add(tf.layers.dense({
        units: 10,
        useBias: true,
        activation: 'sigmoid',
        inputDim: 1,
    }));
    model.add(tf.layers.dense({
        units: 10,
        useBias: true,
        activation: 'sigmoid',
    }));
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'sigmoid',
    }));

    const optimizer = tf.train.adam();
    model.compile({
        loss: 'meanSquaredError',
        optimizer,
    });

    return model;
}

/**
 * Trains the model using the provided training feature and label tensors.
 * @param {tf.Model} model - The TensorFlow.js model to be trained.
 * @param {tf.Tensor} trainingFeatureTensor - The input feature tensor for training.
 * @param {tf.Tensor} trainingLabelTensor - The target label tensor for training.
 * @returns {Promise} A promise that resolves when the model training is complete.
 */
async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {

    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ['loss']
    );

    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 100,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd,
            onEpochBegin: async function () {
                await plotPredictionLine();
                const layer = model.getLayer(undefined, 0);
                tfvis.show.layer({ name: "Layer 1" }, layer);
            }
        }
    });
}


/**
 * Predicts the house price based on the input value.
 * @async
 * @function predict
 */
async function predict() {
    const predictionInput = parseInt(document.getElementById("prediction-input").value);
    if (isNaN(predictionInput)) {
        alert("Please enter a valid number");
    }
    else if (predictionInput < 200) {
        alert("Please enter a value above 200 sqft");
    }
    else {
        tf.tidy(() => {
            const inputTensor = tf.tensor1d([predictionInput]);
            const normalizedInput = normalize(inputTensor, normalizedFeature.min, normalizedFeature.max);
            const normalizedOutputTensor = model.predict(normalizedInput.tensor);
            const outputTensor = denormalize(normalizedOutputTensor, normalizedLabel.min, normalizedLabel.max);
            const outputValue = outputTensor.dataSync()[0];
            const outputValueRounded = (outputValue / 1000).toFixed(0) * 1000;
            document.getElementById("prediction-output").innerHTML = `The predicted house price is <br>`
                + `<span style="font-size: 2em">\$${outputValueRounded}</span>`;
        });
    }
}

const storageID = "kc-house-price-regression";

/**
 * Saves the model to local storage and updates the model status element.
 * @async
 * @function save
 * @returns {Promise<void>} A promise that resolves when the model is saved.
 */
async function save() {
    const saveResults = await model.save(`localstorage://${storageID}`);
    document.getElementById("model-status").innerHTML = `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`;
}

/**
 * Loads a saved model from local storage and performs necessary operations.
 * @returns {Promise<void>} A promise that resolves when the model is loaded and operations are performed.
 */
async function load() {
    const storageKey = `localstorage://${storageID}`;
    const models = await tf.io.listModels();
    const modelInfo = models[storageKey];
    if (modelInfo) {
        model = await tf.loadLayersModel(storageKey);

        tfvis.show.modelSummary({ name: "Model summary" }, model);
        const layer = model.getLayer(undefined, 0);
        tfvis.show.layer({ name: "Layer 1" }, layer);

        await plotPredictionLine();

        document.getElementById("model-status").innerHTML = `Trained (saved ${modelInfo.dateSaved})`;
        document.getElementById("predict-button").removeAttribute("disabled");
    }
    else {
        alert("Could not load: no saved model found");
    }
}

/**
 * Performs a test on the model using the testing feature and label tensors.
 * Prints the testing set loss to the console and updates the testing status element on the page.
 */
async function test() {
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = (await lossTensor.dataSync())[0];
    console.log(`Testing set loss: ${loss}`);

    document.getElementById("testing-status").innerHTML = `Testing set loss: ${loss.toPrecision(5)}`;
}

/**
 * Trains the model by disabling buttons, creating the model, plotting the prediction line,
 * training the model, and updating the model status.
 * @returns {Promise<void>} A promise that resolves when the training is complete.
 */
async function train() {
    // Disable all buttons and update status
    ["train", "test", "load", "predict", "save"].forEach(id => {
        document.getElementById(`${id}-button`).setAttribute("disabled", "disabled");
    });
    document.getElementById("model-status").innerHTML = "Training...";

    const model = createModel();
    tfvis.show.modelSummary({ name: "Model summary" }, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1" }, layer);
    await plotPredictionLine();

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    console.log(result);
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);
    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);

    document.getElementById("model-status").innerHTML = "Trained (unsaved)\n"
        + `Loss: ${trainingLoss.toPrecision(5)}\n`
        + `Validation loss: ${validationLoss.toPrecision(5)}`;
    document.getElementById("test-button").removeAttribute("disabled");
    document.getElementById("save-button").removeAttribute("disabled");
    document.getElementById("predict-button").removeAttribute("disabled");
}

/**
 * Sets the weights of the first layer in the model and plots the prediction line.
 * @param {number} weight - The weight value for the input multiplier.
 * @param {number} bias - The bias value.
 * @returns {Promise<void>} - A promise that resolves when the prediction line is plotted.
 */
async function plotParams(weight, bias) {
    model.getLayer(null, 0).setWeights([
        tf.tensor2d([[weight]]), // Kernel (input multiplier)
        tf.tensor1d([bias]), // Bias
    ])
    await plotPredictionLine();
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1" }, layer);
}

/**
 * Toggles the visibility of the TensorFlow.js visor.
 * @async
 * @function toggleVisor
 * @returns {Promise<void>} A promise that resolves when the visor is toggled.
 */
async function toggleVisor() {
    tfvis.visor().toggle();
}

let points;
let normalizedFeature, normalizedLabel;
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;
async function run() {
    // Ensure backend has initialized
    await tf.ready();

    // Import from CSV
    const houseSalesDataset = tf.data.csv("./kc_house_data.csv");

    // Extract x and y values to plot
    const pointsDataset = houseSalesDataset.map(record => ({
        x: record.sqft_living,
        y: record.price,
    }));
    points = await pointsDataset.toArray();
    if (points.length % 2 !== 0) { // If odd number of elements
        points.pop(); // remove one element
    }
    tf.util.shuffle(points);
    plot(points, "Square feet");

    // Extract Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    // Extract Labels (outputs)
    const labelValues = points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // Normalize features and labels
    normalizedFeature = normalize(featureTensor);
    normalizedLabel = normalize(labelTensor);
    featureTensor.dispose();
    labelTensor.dispose();

    [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2);
    [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2);

    // Update status and enable train button
    document.getElementById("model-status").innerHTML = "No model trained";
    document.getElementById("train-button").removeAttribute("disabled");
    document.getElementById("load-button").removeAttribute("disabled");
}

run();