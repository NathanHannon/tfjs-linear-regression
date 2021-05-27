async function run() {
    const houseSalesDataset = tf.data.csv("http://127.0.0.1:5500/kc_house_data.csv");
    const sampleDataset = houseSalesDataset.take(10);
    const dataArray = await sampleDataset.toArray();

    console.log(dataArray);
}

run();