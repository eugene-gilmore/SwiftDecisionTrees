import Foundation

func insideRule(data : DataSet, rule : Rule) -> DataSet {
    return insideRules(data: data, rules: [(rule: rule, invert: false)])
}

extension Array {
    var combinations: [[Element]] {
        guard !isEmpty else { return [[]] }
        return Array(self[1...]).combinations.flatMap { [$0, [self[0]] + $0] }
    }
}

extension Array where Iterator.Element: FixedWidthInteger {
    var nonzeroBitCount: Int {
        return self.reduce(0, {$0 + $1.nonzeroBitCount})
    }
    
    static func &(lhs: [Iterator.Element], rhs: [Iterator.Element]) -> [Iterator.Element] {
        return lhs.enumerated().map {(index,element) in 
            return element & rhs[index]
        }
    }
}

extension RandomAccessCollection {
    func binarySearch(predicate: (Iterator.Element) -> Bool) -> Index? {
        var low = startIndex
        var high = endIndex
        guard predicate(self[low]), !predicate(self[index(before: high)]) else {
            return nil
        }
        while low != high {
            let mid = index(low, offsetBy: distance(from: low, to: high)/2)
            if predicate(self[mid]) {
                low = index(after: mid)
            } else {
                high = mid
            }
        }
        return low
    }
}

func impurity(data : DataSet, forClassValue: Int) -> Int {
    var numOther : Int = 0
    for p in 0..<data.instances.count {
        if(data.instances[p].classVal != forClassValue) {
            numOther += 1
        }
    }
    return numOther
}

func impurity(data : DataSet, rule : Rule, forClassValue: Int) -> Int {
    let inside = insideRule(data: data, rule: rule)
    return impurity(data: inside, forClassValue: forClassValue)
}

func impurityBits(data: DataSet, rule: Rule, forClassValue: Int) -> [Int64] {
    var result = Array<Int64>(repeating: 0, count: Int(ceil(Double(data.instances.count)/64.0)))
    for p in 0..<data.instances.count {
        if(data.instances[p].classVal != forClassValue) {
            let inside = insideRule(instance: data.instances[p], data: data, rule: rule)
            if(inside == nil || inside == true) {
                result[p/64] = result[p/64] | (Int64(1) << (p%64))
            }
        }
    }
    return result
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
                if(attMin == attMax || (abs(r.min - attMin) < tolerance && abs(r.max - attMax) < tolerance)) {
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
            var newMin : Double? = nil
            var newMax : Double? = nil
            if let min = s.rangeMin,
                let index = data.instances.binarySearch(predicate: {$0.values[s.axisIndex] ?? Double.infinity < min}),
                let v = data.instances[index-1].values[s.axisIndex] {
                    newMin = v+(min-v)/2.0
            }
            if let max = s.rangeMax,
                let index = data.instances.binarySearch(predicate: {$0.values[s.axisIndex] ?? Double.infinity <= max}), index != data.instances.startIndex,
                let v = data.instances[index].values[s.axisIndex] {
                newMax = max+(v-max)/2.0
            }
            if(newMin != nil || newMax != nil) {
                result.append(AxisSelectionRule(rangeMin: newMin, rangeMax: newMax, axisIndex: s.axisIndex))
            }
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

    var allRulesImpurity = allRules.map { r in 
        return (rule : r, impurityBits: impurityBits(data: data, rule: Rule.AxisSelection([r]), forClassValue: forClassValue))
    }.sorted {$0.impurityBits.nonzeroBitCount < $1.impurityBits.nonzeroBitCount}
    
    let first = allRulesImpurity.removeFirst()
    var combination : (rule: [AxisSelectionRule], impurityBits: [Int64]) = ([first.rule], first.impurityBits) 
    while(!allRulesImpurity.isEmpty && 
    combination.impurityBits.nonzeroBitCount > smallestNumOther) {
        let next = allRulesImpurity.removeFirst()
        combination.rule.append(next.rule)
        combination.impurityBits = combination.impurityBits & next.impurityBits
    }
    return correctDecisionBoundaries(forRule: Rule.AxisSelection(combination.rule), data: data)


    /*var combinations = allRulesImpurity.combinations.sorted {$0.count < $1.count }
    combinations.removeFirst()
    combinations.removeLast()
    for c in combinations {
        let impurity = c.reduce(c[0].impurityBits, {$0 & $1.impurityBits})
        if(impurity.nonzeroBitCount <= smallestNumOther) {
            let r = Rule.AxisSelection(c.map {$0.rule})
            return correctDecisionBoundaries(forRule: r, data: data)
        }
    }
    return correctDecisionBoundaries(forRule: Rule.AxisSelection(allRules), data: data)*/
}

public func findBestCavity(data : DataSet) -> Rule? {
    var lowestNumOther : Int? = nil
    var bestClass : Int? = nil
    let dist = distribution(data: data)
    for c in data.classes {
        guard let count = dist[c.value], count >= 2 else {
            continue
        }
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

public func findBestCavityC45(data : DataSet) -> Rule? {
    let c45 = findBestSplit(data: data, twoValueSplit: false, j48Mode: true)
    if let cavity = findBestCavity(data: data) {
        let cavityGainRatio = gainRatio(distribution: Distribution(dataset: data, rule: cavity))
        if let c45Rule = c45.rule {
            if(c45.gainRatio > cavityGainRatio) {
                return c45Rule
            }
        }
        return cavity
    }
    return c45.rule
}
