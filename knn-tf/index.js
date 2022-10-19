//require('@tensorflow/tfjs-node'); // tfjs-node instruct Tensorflow where to do calculations. This way it will be on CPU.
// When importing TensorFlow.js from this package, the module that you get will be accelerated by the TensorFlow C binary and run on the CPU

const tf = require('@tensorflow/tfjs'); // well here it will use GPU by defaule since
// I commented require('@tensorflow/tfjs-node'); because it was giving me errors
const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10, // this will set aside 10 set of data for testing purposes.
    dataColumns: ['lat', 'long'],
    labelColumns: ['price']
})

console.log(testFeatures)
console.log(testLabels)