async function plot(pointsArray, featureName) {
    tfvis.render.scatterplot(
        { name: `${featureName} vs House Price` },
        { values: [pointsArray], series: ["original"] },
        {
            xLabel: featureName,
            yLabel: "Price",
        }
    );
}

function normalize(tensor) {
    const min = tensor.min();
    const max = tensor.max();
    const normalizedTensor = tensor.sub(min).div(max.sub(min));
    return {
        tensor: normalizedTensor,
        min,
        max
    }
}

function denormalize(tensor, min, max) {
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
    return denormalizedTensor;
}

function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({

    }));

    return model;
}

async function run() {
    // Import from CSV
    const houseSalesDataset = tf.data.csv("./kc_house_data.csv");

    // Extract X and Y values from dataset and plot
    const pointsDataset = houseSalesDataset.map(record => ({
        x: record.sqft_living,
        y: record.price,
    }));
    const points = await pointsDataset.toArray();
    tf.util.shuffle(points);
    plot(points, "Square Feet");

    // Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    // Labels (outputs)
    const labelValues = points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // Normalize features (inputs) and labels (outputs)
    const normalizedFeature = normalize(featureTensor);
    const normalizedLabel = normalize(labelTensor);

    const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2);
    const [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2);

    const model = createModel();
}

run();