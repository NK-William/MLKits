//require('@tensorflow/tfjs-node'); // tfjs-node instruct Tensorflow where to do calculations. This way it will be on CPU.
// When importing TensorFlow.js from this package, the module that you get will be accelerated by the TensorFlow C binary and run on the CPU

const tf = require('@tensorflow/tfjs'); // well here it will use GPU by defaule since
// I commented require('@tensorflow/tfjs-node'); because it was giving me errors
const loadCSV = require('./load-csv');

function knn(features, labels, predictionPoint, k) {
return features
    .sub(predictionPoint)
    .pow(2)
    .sum(1)
    .pow(.5) // hypotenuse now complete 
    .expandDims(1)
    .concat(labels, 1)
    .unstack()  // separates a tensor by rows of tensors inside a JS array
    .sort((a, b) => a.get(0) > b.get(0)? 1 : -1) // using a JS operation since we have JS array of tensors
    .slice(0, k) // JS version of slice
    .reduce((acc, pair) => acc + pair.get(1), 0) / k // JS version of reduce
}

let { features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10, // this will set aside 10 set of data for testing purposes.
    dataColumns: ['lat', 'long'],
    labelColumns: ['price']
})

features = tf.tensor(features);
labels = tf.tensor(labels);

// const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);
// console.log('Guess', result, testLabels[0][0]);

const ts = tf.tensor([1,2,3,4,5,6]);
console.log("value: ", ts.get(0));