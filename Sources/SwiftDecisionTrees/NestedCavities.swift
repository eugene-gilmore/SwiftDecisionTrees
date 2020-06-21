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

public func findNextCavity(forClassValue : Int, data : DataSet) -> Rule? {
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
    return Rule.AxisSelection(allRules)
}
