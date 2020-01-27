//
//  DecisionList.swift
//  Classifier-Builder
//
//  Created by Eugene Gilmore on 17/6/17.
//
//

typealias DecisionListEntry = (rules: [AxisSelectionRule], classValue : Int)

class DecisionList {
	public var entries : [DecisionListEntry]
	private var dataSet : DataSet
	
	public init? (fromDecisionTree : TreeNode, dataset : DataSet) {
		func traverseTree(t : TreeNode, rules : [AxisSelectionRule]) -> [DecisionListEntry]? {
			if(t.isLeaf()) {
				return [DecisionListEntry(rules: rules, classValue: t.classVal!)]
			}
			else if(t.hasChildren()){
				switch t.rules! {
				case var .AxisSelection(s):
                    for r in 0..<s.count {
                        if(s[r].rangeMax == nil) {
                            s[r].rangeMax = dataset.attributes[s[r].axisIndex].max!
                        }
                        if(s[r].rangeMin == nil) {
                            s[r].rangeMin = dataset.attributes[s[r].axisIndex].min!
                        }
                    }
					let l1 = traverseTree(t: t.insideChildRule!, rules: rules + s)
					let l2 = traverseTree(t: t.outsideChildRule!, rules: rules)
					if(l1 == nil || l2 == nil) {
						return nil
					}
					return l1! + l2!
				default:
					print("Unsupported Rule Type!")
					return nil
				}
			}
			else {
				print("Warning Incomplete Tree!")
				return nil
			}
		}
		if let e = traverseTree(t: fromDecisionTree, rules: []) {
			entries = e
		}
		else {
			return nil
		}
		self.dataSet = dataset
	}
	
	public func saveToFile(file : String, precision : Int) {
		var dlc = ""
		for i in entries {
			dlc += "\(i.classValue)"
			for j in dataSet.attributes.indices {
				if let r = i.rules.first(where: { $0.axisIndex == j }) {
                    var min = 0.0, max = 0.0
                    if let m = r.rangeMax {
                        min = m
                    }
                    else {
                        min = dataSet.attributes[j].min!
                    }
                    if let m = r.rangeMax {
                        max = m
                    }
                    else {
                        max = dataSet.attributes[j].max!
                    }
					dlc += String(format: "%.\(precision)f %.\(precision)f ", min, max)
				}
				else {
					dlc += String(format: "%.\(precision)f %.\(precision)f ", dataSet.attributes[j].min!, dataSet.attributes[j].max!)
				}
			}
		}
		do {
			try dlc.write(toFile: file, atomically: false, encoding: .utf8)
		}
		catch {
			print("Error writing File")
		}
	}
	
	public func simplifyRule(dle : DecisionListEntry) -> DecisionListEntry? {
		var dle = dle
		for i in 0..<dle.rules.count {
			for j in dle.rules.indices.dropFirst(i+1).reversed() {
				if dle.rules[i].axisIndex == dle.rules[j].axisIndex {
					if(dle.rules[j].rangeMax! < dle.rules[i].rangeMin! || dle.rules[j].rangeMin! > dle.rules[i].rangeMax!) {
						return nil //no overlap in range for same attribute for this entry so this combination of rules will always be false
					}
					//take the intersection
					dle.rules[i].rangeMin = max(dle.rules[i].rangeMin!, dle.rules[j].rangeMin!)
					dle.rules[i].rangeMax = min(dle.rules[i].rangeMax!, dle.rules[j].rangeMax!)
					dle.rules.remove(at: j)
				}
			}
		}
		return dle
	}
	
	public func simplifyRules() {
		for i in entries.indices.reversed() {
			if let x = simplifyRule(dle: entries[i]) {
				entries[i] = x
			}
			else {
				entries.remove(at: i)
			}
		}
	}
}
