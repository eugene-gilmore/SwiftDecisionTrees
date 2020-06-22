func insideRule(data : DataSet, rule : Rule) -> DataSet {
    return insideRules(data: data, rules: [(rule: rule, invert: false)])
}

extension Array {
    var combinations: [[Element]] {
        guard !isEmpty else { return [[]] }
        return Array(self[1...]).combinations.flatMap { [$0, [self[0]] + $0] }
    }
}

func impurity(data : DataSet, forClassValue: Int) -> Double {
    var numOther : Double = 0
    for p in 0..<data.instances.count {
        if(data.instances[p].classVal != forClassValue) {
            numOther += data.weights[p] ?? 1.0
        }
    }
    return numOther
}

func impurity(data : DataSet, rule : Rule, forClassValue: Int) -> Double {
    let inside = insideRule(data: data, rule: rule)
    return impurity(data: inside, forClassValue: forClassValue)
}

func findAllRules(forClassValue : Int, data : DataSet) -> [AxisSelectionRule]? {
    let classValueRanges = data.instances.reduce(
        Array<(min: Double, max: Double)?>(repeating: nil, count: data.numAttributes()), 
        {(prev: [(min: Double, max: Double)?], p : Point) in 
            var result = prev
            if(p.classVal != forClassValue) {
                return result
            }
            for i in 0..<p.values.count {
                if let v = p.values[i] {
                    if(result[i] == nil) {
                        result[i] = (min: v, max: v)
                    }
                    else {
                        result[i] = (min: min(result[i]!.min, v), max: max(result[i]!.max, v))
                    }
                }
            }
            return result
        }
    )
    let allRules : [AxisSelectionRule] = classValueRanges.enumerated().map( { (index,range) in 
        if let r = range {
            let att = data.getAttributes()[index]
            if let attMin = att.min, let attMax = att.max {
                let tolerance = (attMax - attMin)/1000.0
                if(abs(r.min - attMin) < tolerance && abs(r.max - attMax) < tolerance) {
                    return nil
                }
            }
            return AxisSelectionRule(rangeMin: r.min, rangeMax: r.max, axisIndex: index)
        }
        return nil
    }).compactMap{$0}

    if(allRules.isEmpty) {
        return nil
    }
    return allRules
}

func correctDecisionBoundaries(forRule rule : Rule, data : DataSet) -> Rule {
    switch rule {
    case let .AxisSelection(selections):
        var result : [AxisSelectionRule] = []
        for s in selections {
            let _ = data.sortOnAttribute(attribute: s.axisIndex)
            var index = 0
            var newMin : Double? = nil
            var newMax : Double? = nil
            if let min = s.rangeMin {
                while(data.instances[index].values[s.axisIndex] ?? Double.infinity < min) {
                    index += 1
                    if(index >= data.instances.count) {
                        index -= 1
                        break
                    }
                }
                if let v = data.instances[index].values[s.axisIndex], v < min {
                    newMin = v+(v+min)/2.0
                }
            }
            if let max = s.rangeMax {
                while(index < data.instances.count && !(data.instances[index].values[s.axisIndex] ?? Double.infinity > max)) {
                    index += 1
                    if(index >= data.instances.count) {
                        index -= 1
                        break
                    }
                }
                if let v = data.instances[index].values[s.axisIndex], v > max {
                    newMax = v+(v+max)/2.0
                }
            }
            result.append(AxisSelectionRule(rangeMin: newMin, rangeMax: newMax, axisIndex: s.axisIndex))
        }
        return Rule.AxisSelection(result)
    default:
        return rule
    }
}

public func findNextCavity(forClassValue : Int, data : DataSet) -> Rule? {
    
    guard let allRules = findAllRules(forClassValue: forClassValue, data: data) else {
        return nil
    }

    let smallestNumOther = impurity(data: data, rule: Rule.AxisSelection(allRules), forClassValue: forClassValue)


    var combinations = allRules.combinations.sorted {$0.count < $1.count }
    combinations.removeFirst()
    combinations.removeLast()
    for c in combinations {
        let r = Rule.AxisSelection(c)
        if(impurity(data: data, rule: r, forClassValue: forClassValue) - 0.1 < smallestNumOther) {
            return r
        }
    }
    return correctDecisionBoundaries(forRule: Rule.AxisSelection(allRules), data: data)
}

public func findBestCavity(data : DataSet) -> Rule? {
    var lowestNumOther : Double? = nil
    var bestClass : Int? = nil
    for c in data.classes {
        if let allRules = findAllRules(forClassValue: c.value, data: data) {
            let numOther = impurity(data: data, rule: Rule.AxisSelection(allRules), forClassValue: c.value)
            if(lowestNumOther == nil || lowestNumOther! > numOther) {
                lowestNumOther = numOther
                bestClass = c.value
            }
        }
    }
    if let c = bestClass {
        return findNextCavity(forClassValue: c, data: data)
    }
    return nil
}