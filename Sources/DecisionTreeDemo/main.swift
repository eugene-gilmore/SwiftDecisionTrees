import MachineLearning
import Foundation
import Signals

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
let resultsLocation = datasetLocation+"results/oc1/"

//generateMostlyRandom(numAttributes: 1000, numNonRandom: 5, numInstances: 200).saveAsCSV(file: datasetLocation+"generated1.csv")

let datasets = ["iris", "liver", "cryotherapy", "seeds", "ecoli", "car", "breast-cancer-wisconsin", "glass", "vowel", "page-blocks", "wine", "heart", "credit", "vehicle", "ionosphere",
    "ClimateSimulationCrashes", "leaf", "chronic_kidney_disease", "BreastTissue", "transfusion"]
let NUM_RUNS = 5

func printDatasetStats(datasets : [String]) {
    for d in datasets {
        guard let data = loadFromFile(file: datasetLocation+d+".csv", headingsPresent: false) else {
            print("couldn't load file \(datasetLocation+d+".csv")")
            continue
        }
        print("\(d)\t\(data.instances.count)\t\(data.attributes.count)\t\(data.classes.count)")
    }
}

//printDatasetStats(datasets: datasets)

let progress = CrossValidationProgress()

Signals.trap(signal: Signals.Signal.user(Int(SIGUSR1))) { _ in
    let p = progress.progress()
    print("\(String(format: "%.3f", p.complete*100.0))% of run complete in \(String(format: "%.1f seconds", p.time))")
}

//for d in datasets {
//    guard let data = loadFromFile(file: datasetLocation+d+".csv", headingsPresent: false) else {
//        print("couldn't load file \(datasetLocation+d+".csv")")
//        continue
//    }
//    data.saveAsARFF(file: datasetLocation+"generatedARFF/\(d).arff")
//}
//

func processWeka(directory: String) {
    for d in datasets {
        print(d)
        var avg = 0.0
        for r in 0..<NUM_RUNS {
            if let result = try? Result(filename: directory+d+"-Run\(r+1)-Result") {
                //print("Run \(r+1) \(result.macroFMeasureV2())")
                avg += result.macroFMeasureV2()
            }
            else {
                print("Error reading file " + directory+d+"-Run\(r+1)-Result")
            }
        }
        print(String(format: "%.2f", avg/Double(NUM_RUNS)))
    }
}

func debug() {
    func average(a : [Double]) -> Double {
        return a.reduce(0.0, {$0 + $1})/Double(a.count)
    }
    func resultString(run: Int?, fMeasure: Double, treeSize: Double, deepestLeaf: Double) -> String {
        return "\(run != nil ? "Run \(run!+1)" : "Average")\t\(String(format: "%.2f",fMeasure))\t\t\(String(format: "%.2f",treeSize))\t\t\(String(format: "%.2f",deepestLeaf))\n"
    }
    let start = Date().timeIntervalSince1970
    var result : [Result] = []
    var aggregateResult = Result()
    aggregateResult.confusionMatrix = Array(repeating: Array(repeating: 0, count: 2), count: 2)
    var fMeasure : [Double] = []
    var treeSize : [Double] = []
    var deepestLeaf : [Double] = []
    var resultStr = ""
    for k in 0..<10 {
        if let r = try? Result(filename: resultsLocation+"../breast-cancer-wisconsin-Run1-Result\(k)") {
            result.append(r)
            for i in 0..<2 {
                for j in 0..<2 {
                    aggregateResult.confusionMatrix[i][j] += result[k].confusionMatrix[i][j]
                }
            }
        }
        else {
            print("Error reading file")
        }
    }
    fMeasure.append(aggregateResult.macroFMeasureV2())
    treeSize.append(average(a: result.map {Double($0.tree.sizeOfTree())}))
    deepestLeaf.append(average(a: result.map {Double($0.tree.deepestLeaf())}))
    resultStr += resultString(run: 0, fMeasure: fMeasure.last!, treeSize: treeSize.last!, deepestLeaf: deepestLeaf.last!)
    print("Run \(0+1)/\(NUM_RUNS) Complete(\(String(format: "%.1f", (Date().timeIntervalSince1970-start)))s)")
    
}

func printDatasetStatsLatex() {
    print("Dataset & Attributes & Instances & Classes \\\\")
    for d in datasets {
        guard let data = loadFromFile(file: datasetLocation+d+".csv", headingsPresent: false) else {
            print("couldn't load file \(datasetLocation+d+".csv")")
            continue
        }
        print("\(d) & \(data.attributes.count) & \(data.instances.count) & \(data.classes.count)\\\\")
    }
}

//printDatasetStatsLatex()

//debug()

//processWeka(directory: resultsLocation+"../j48/")

func evaluateSize() {
    for d in datasets {
        guard let data = loadFromFile(file: datasetLocation+d+".csv", headingsPresent: false) else {
            print("couldn't load file \(datasetLocation+d+".csv")")
            continue
        }
        var size = 0.0
        var deepestLeaft = 0.0
        for _ in 0..<NUM_RUNS {
            let tree = TreeNode()
            finishSubTree(node: tree, data: data, fullTrainingSet: data, buildMethod: .OC1)
            size += Double(tree.sizeOfTree())
            deepestLeaft += Double(tree.deepestLeaf())
        }
        print("\(d) \(deepestLeaft/Double(NUM_RUNS))\\\(size/Double(NUM_RUNS))")
    }
}

//evaluateSize()
//exit(0)

for d in datasets {
    guard let data = loadFromFile(file: datasetLocation+d+".csv", headingsPresent: false) else {
        print("couldn't load file \(datasetLocation+d+".csv")")
        continue
    }
    print("loaded: \(datasetLocation+d+".csv")")
    var fMeasure : [Double] = []
    var accuracy: [Double] = []
    var treeSize : [Double] = []
    var deepestLeaf : [Double] = []
    var resultStr = "\tFMeasure\tTreeSize\tDeepestLeaf\n"
    func resultString(run: Int?, fMeasure: Double, accuracy: Double, treeSize: Double, deepestLeaf: Double) -> String {
        return "\(run != nil ? "Run \(run!+1)" : "Average")\t\(String(format: "%.2f",fMeasure))\t\t\(String(format: "%.2f", accuracy))\t\t\(String(format: "%.2f",treeSize))\t\t\(String(format: "%.2f",deepestLeaf))\n"
    }
    func average(a : [Double]) -> Double {
        return a.reduce(0.0, {$0 + $1})/Double(a.count)
    }
    for i in 0..<NUM_RUNS {
        let start = Date().timeIntervalSince1970
        let result = crossValidation(data: data, buildMethod: .DE, progress: progress, runParallel: false)
        var aggregateResult = Result()
        aggregateResult.confusionMatrix = Array(repeating: Array(repeating: 0, count: data.classes.count), count: data.classes.count)
        for r in 0..<result.count {
            result[r].tree.saveToFile(filename: resultsLocation+d+"-Run\(i+1)-Tree\(r)")
            result[r].saveToFile(filename: resultsLocation+d+"-Run\(i+1)-Result\(r)")
            result[r].testSet.saveAsCSV(file: resultsLocation+d+"-Run\(i+1)-TestSet\(r)")
            for i in 0..<data.classes.count {
                for j in 0..<data.classes.count {
                    aggregateResult.confusionMatrix[i][j] += result[r].confusionMatrix[i][j]
                }
            }
        }
        fMeasure.append(aggregateResult.macroFMeasureV2())
        accuracy.append(aggregateResult.accuracy())
        treeSize.append(average(a: result.map {Double($0.tree.sizeOfTree())}))
        deepestLeaf.append(average(a: result.map {Double($0.tree.deepestLeaf())}))
        resultStr += resultString(run: i, fMeasure: fMeasure.last!, accuracy: accuracy.last!, treeSize: treeSize.last!, deepestLeaf: deepestLeaf.last!)
        print("Run \(i+1)/\(NUM_RUNS) Complete(\(String(format: "%.1f", (Date().timeIntervalSince1970-start)))s)")
    }
    resultStr += resultString(run: nil, fMeasure: average(a: fMeasure), accuracy: average(a: accuracy), treeSize: average(a: treeSize), deepestLeaf: average(a: deepestLeaf))
    do {
        try resultStr.write(toFile: resultsLocation+d+"-results.txt", atomically: false, encoding: .utf8)
    }
    catch {
        print("Error writting results to file")
    }
}
