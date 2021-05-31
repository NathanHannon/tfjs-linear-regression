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

async function run() {
    // Import from CSV
    const houseSalesDataset = tf.data.csv("http://127.0.0.1:5500/kc_house_data.csv");

    // Extract X and Y values from dataset and plot
    const points = houseSalesDataset.map(record => ({
        x: record.sqft_living,
        y: record.price,
    }));
    plot(await points.toArray(), "Square Feet");

    // Features (inputs)
    const featureValues = await points.map(p => p.x).toArray();
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    // Labels (outputs)
    const labelValues = await points.map(p => p.y).toArray();
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // Normalize features (inputs) and labels (outputs)
    const normalizedFeature = normalize(featureTensor);
    const normalizedLabel = normalize(labelTensor);

}

run();