import MachineLearning

let datasetLocation = "/Users/eugene/PHD/Datasets/"
let datasets = ["iris"]

for d in datasets {
    guard let data = loadFromFile(file: datasetLocation+d+".csv") else {
        continue
    }
    let result = crossValidation(data: data, buildMethod: .DE)
    let fMeasure = result.reduce(0.0, {$0 + $1.macroFMeasure()})/Double(result.count)
    print(fMeasure)
}
