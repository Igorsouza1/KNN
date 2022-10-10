const treinox = require("./novasMedidasX")
const treinoy = require("./novasMedidasY")
const algorith = require("machinelearn/neighbors")
const crossValidation = require('ml-cross-validation');
const KNN = require('ml-knn')

const confusionMatrix = crossValidation.kFold(treinox, treinoy,5, function(trainFeatures, trainLabels, testFeatures) {
  var knn = new KNN(trainFeatures, trainLabels, {k: 3});
  return knn.predict(testFeatures);
});

const accuracy = confusionMatrix.getAccuracy();
console.log(confusionMatrix)
console.log(accuracy)