import MachineLearning
import Foundation

//taken from https://stackoverflow.com/questions/49470358/how-can-i-return-a-float-or-double-from-normal-distribution-in-swift-4
class MyGaussianDistribution {
    let mean: Double
    let deviation: Double

    init(mean: Double, deviation: Double) {
        precondition(deviation >= 0)
        self.mean = mean
        self.deviation = deviation
    }

    func nextDouble() -> Double {
        guard deviation > 0 else { return mean }

        let x1 = Double.random(in: 0.0...1.0) // a random number between 0 and 1
        let x2 = Double.random(in: 0.0...1.0) // a random number between 0 and 1
        let z1 = sqrt(-2 * log(x1)) * cos(2 * Double.pi * x2) // z1 is normally distributed

        // Convert z1 from the Standard Normal Distribution to our Normal Distribution
        return z1 * deviation + mean
    }
}

func generateMostlyRandom(numAttributes: Int, numNonRandom: Int, numInstances: Int) -> DataSet {
    var numNonRandom = numNonRandom
    if(numNonRandom > numAttributes) {
        numNonRandom = numAttributes
    }
    let result = DataSet()
    for i in 0..<numAttributes {
        result.addAttribute(name: "\(i)")
    }
    
    let classVals = (1...numInstances).map( { _ in Int.random(in: 0...1) } )
    
    let nonRandom = (0..<numAttributes).shuffled().prefix(numNonRandom)
    var nonRandomValues : [[Double]] = []
    for _ in 0..<numNonRandom {
        var vals : [Double] = []
        let selectedClass = Int.random(in: 0...1)
        let dist = MyGaussianDistribution(mean: Double.random(in: 0...1.0), deviation: Double.random(in: 0.05...0.08))
        for j in 0..<numInstances {
            if(classVals[j] == selectedClass) {
                vals.append(max(0.0, min(1.0, dist.nextDouble())))
            }
            else {
                vals.append(Double.random(in: 0...1.0))
            }
        }
        nonRandomValues.append(vals)
    }
    
    for i in 0..<numInstances {
        var vals : [Double] = []
        for j in 0..<numAttributes {
            if let index = nonRandom.firstIndex(of: j) {
                vals.append(nonRandomValues[index][i])
            }
            else {
                vals.append(Double.random(in: 0...1.0))
            }
        }
        result.addPoint(point: Point(values: vals, classVal: classVals[i]))
    }
    return result
}

let datasetLocation = "/Users/eugene/PHD/Datasets/"
let resultsLocation = datasetLocation+"results/c45/"

//generateMostlyRandom(numAttributes: 1000, numNonRandom: 5, numInstances: 200).saveAsCSV(file: datasetLocation+"generated1.csv")

let datasets = ["iris", "breast-cancer-wisconsin", "glass", "vehicle", "vowel", "heart", "credit", "wine", "ionosphere", "car", "liver", "page-blocks", "ecoli", "seeds", "cryotherapy"]
let NUM_RUNS = 5

for d in datasets {
    guard let data = loadFromFile(file: datasetLocation+d+".csv") else {
        print("couldn't load file \(datasetLocation+d+".csv")")
        continue
    }
    print("loaded: \(datasetLocation+d+".csv")")
    var fMeasure : [Double] = []
    var treeSize : [Double] = []
    var deepestLeaf : [Double] = []
    var resultStr = "\tFMeasure\tTreeSize\tDeepestLeaf\n"
    func resultString(run: Int?, fMeasure: Double, treeSize: Double, deepestLeaf: Double) -> String {
        return "\(run != nil ? "Run \(run!+1)" : "Average")\t\(String(format: "%.2f",fMeasure))\t\t\(String(format: "%.2f",treeSize))\t\t\(String(format: "%.2f",deepestLeaf))\n"
    }
    func average(a : [Double]) -> Double {
        return a.reduce(0.0, {$0 + $1})/Double(a.count)
    }
    for i in 0..<NUM_RUNS {
        let start = Date().timeIntervalSince1970
        let result = crossValidation(data: data, buildMethod: .C45)
        fMeasure.append(average(a: result.map {$0.macroFMeasure()}))
        treeSize.append(average(a: result.map {Double($0.tree.sizeOfTree())}))
        deepestLeaf.append(average(a: result.map {Double($0.tree.deepestLeaf())}))
        resultStr += resultString(run: i, fMeasure: fMeasure.last!, treeSize: treeSize.last!, deepestLeaf: deepestLeaf.last!)
        for r in 0..<result.count {
            result[r].tree.saveToFile(filename: resultsLocation+d+"-Run\(i+1)-Tree\(r)")
        }
        print("Run \(i+1)/\(NUM_RUNS) Complete(\(String(format: "%.1f", (Date().timeIntervalSince1970-start)))s)")
    }
    resultStr += resultString(run: nil, fMeasure: average(a: fMeasure), treeSize: average(a: treeSize), deepestLeaf: average(a: deepestLeaf))
    do {
        try resultStr.write(toFile: resultsLocation+d+"-results.txt", atomically: false, encoding: .utf8)
    }
    catch {
        print("Error writting results to file")
    }
}
