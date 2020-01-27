//
//  HillClimber.swift
//  ClassifierBuilder
//
//  Created by Eugene Gilmore on 9/10/19.
//

protocol HillClimber {
    func EvaluateCost(parameters : [Double]) -> Double
    func NumberOfParameters() -> Int
    func GetConstraints() -> [(min : Double, max : Double)]
    mutating func GetInitialCandidate() -> [Double]
}

extension HillClimber {
    mutating func Optimize(maxIterations : Int, stepSize : Double) -> [Double] {
        var currentCandidate = GetInitialCandidate()
        let constraints = GetConstraints()
        if(currentCandidate.count < NumberOfParameters()) {
            for i in currentCandidate.count..<NumberOfParameters() {
                currentCandidate.append(Double.random(in: constraints[i].min...constraints[i].max))
            }
        }
        var currentCost = EvaluateCost(parameters: currentCandidate)
        
        for iterations in 0..<maxIterations {
            var updated = false
            for i in 0..<NumberOfParameters() {
                for direction in [-1.0,1.0] {
                    var newCandidate = currentCandidate
                    newCandidate[i] += stepSize*direction
                    let newCost = EvaluateCost(parameters: newCandidate)
                    if(newCost < currentCost) {
                        currentCandidate = newCandidate
                        currentCost = newCost
                        updated = true
                        break
                    }
                }
                if(updated) {
                    break
                }
            }
            if(!updated) {
                break
            }
        }
        return currentCandidate
    }
}
