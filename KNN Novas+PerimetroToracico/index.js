const treinox = require("./novasMedidasPerimetroToracicoX")
const treinoy = require("./novasMedidasPerimetroToracicoY")
const algorith = require("machinelearn/neighbors")
const crossValidation = require('ml-cross-validation');


const confusionMatrix = crossValidation.kFold(treinox, treinoy,20, function(trainFeatures, trainLabels, testFeatures) {
  const knn = new algorith.KNeighborsClassifier({k: 3});
  knn.fit(trainFeatures ,trainLabels);
  return knn.predict(testFeatures);
});

const accuracy = confusionMatrix.getAccuracy();
console.log(confusionMatrix)
console.log(accuracy)