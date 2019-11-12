//
//  DifferentialEvolution.swift
//  Classifier-Builder
//
//  Created by Eugene Gilmore on 20/12/18.
//

import Foundation

protocol DifferentialEvolution {
    func EvaluateCost(parameters : [Double]) -> Double
    func NumberOfParameters() -> Int
    func GetConstraints() -> [(min : Double, max : Double)]
    mutating func GetInitialCandidate() -> [Double]
}

extension DifferentialEvolution {
    mutating func Optimize(iterations : Int, populationSize : Int, mutationFactor : Double, crossoverFactor : Double, percentageInitialProvided : Double, onIterationComplete : (() -> Void)? = nil) -> [Double] {
        var population = [[Double]](repeating: [Double](repeating: 0, count: NumberOfParameters()), count: populationSize)
        let constraints = GetConstraints()
        var minCostPerAgent = [Double](repeating: 0, count: populationSize)
        var minCost = Double.infinity
        var bestAgentIndex = 0
        
        var randomPop = 0
        if(percentageInitialProvided > 0.0 && percentageInitialProvided <= 1.0) {
            randomPop = Int(percentageInitialProvided*Double(populationSize))
        }
        for p in 0..<randomPop {
            population[p] = GetInitialCandidate()
            if(population[p].count != NumberOfParameters()) {
                population[p] = Array(repeating: 0, count: NumberOfParameters())
                for i in 0..<NumberOfParameters() {
                    population[p][i] = Double.random(in: constraints[i].min...constraints[i].max)
                }
            }
            minCostPerAgent[p] = EvaluateCost(parameters: population[p])
        }
        
        for p in randomPop..<populationSize {
            for i in 0..<NumberOfParameters() {
                population[p][i] = Double.random(in: constraints[i].min...constraints[i].max)
            }
            minCostPerAgent[p] = EvaluateCost(parameters: population[p])
        }
        for p in 0..<populationSize {
            if(minCostPerAgent[p] < minCost) {
                minCost = minCostPerAgent[p]
                bestAgentIndex = p
            }
        }
        
        for _ in 0..<iterations {
            var x = 0
            while x<populationSize {
                var a = x
                var b = x
                var c = x
                
                while(a == x || b == x || c == x || a == b || a == c || b == c) {
                    a = Int.random(in: 0..<populationSize)
                    b = Int.random(in: 0..<populationSize)
                    c = Int.random(in: 0..<populationSize)
                }
                
                var z = population[a]
                for j in 0..<NumberOfParameters() {
                    z[j] += mutationFactor*(population[b][j] - population[c][j])
                }
                
                let R = Int.random(in: 0..<NumberOfParameters())
                for j in 0..<NumberOfParameters() {
                    if(Double.random(in: 0...1) < crossoverFactor || j == R) {
                        //Keep new agent
                    }
                    else {
                        z[j] = population[x][j]
                    }
                }
                
                var shouldRedo = false
                for j in 0..<NumberOfParameters() {
                    if(z[j] < constraints[j].min || z[j] > constraints[j].max) {
                        shouldRedo = true
                        break
                    }
                }
                if(shouldRedo) {
                    continue
                }
                
                let newCost = EvaluateCost(parameters: z)
                if(newCost < minCostPerAgent[x]) {
                    population[x] = z
                    minCostPerAgent[x] = newCost
                }
                
                if(minCostPerAgent[x] < minCost) {
                    minCost = minCostPerAgent[x]
                    bestAgentIndex = x
                }
                x += 1
            }
            if let complete = onIterationComplete {
                complete()
            }
            //print("Current cost \(minCost)")
        }
        
        return population[bestAgentIndex]
    }
}


struct Rastrigin : HillClimber {
    func GetInitialCandidate() -> [Double] {
        return []
    }
    
    func EvaluateCost(parameters: [Double]) -> Double {
        let A : Double = 10
        var val : Double = 0
        
        for i in 0..<NumberOfParameters() {
            if(parameters[i] < -5.12 || parameters[i] > 5.12) {
                return 1e7
            }
            val += parameters[i]*parameters[i] - A*cos(2*Double.pi*parameters[i])
        }
        
        return A * Double(NumberOfParameters()) + val
    }
    
    func NumberOfParameters() -> Int {
        return 1
    }
    
    func GetConstraints() -> [(min: Double, max: Double)] {
        return [(Double,Double)](repeating: (-5.12,5.12), count: NumberOfParameters())
    }
}
