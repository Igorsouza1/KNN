const { PCA } = require('ml-pca');
const dataset = require('./todasMedidas')
const { normalize }  = require('machinelearn/preprocessing')


const result = normalize(dataset, { norm: 'l2' });
const pca = new PCA(result);
console.log(pca.getExplainedVariance());



