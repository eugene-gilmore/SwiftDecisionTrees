//
//  DecisionTree.swift
//  ClassifierVisualiser
//
//  Created by Eugene Gilmore on 17/11/16.
//
//

import JSONCodable
import Foundation
import OC1

public struct Result : JSONCodable {
    public init(object: JSONObject) throws {
        let decoder = JSONDecoder(object: object)
        let tmp : [Int] = try decoder.decode("confusionMatrix")
        let numClasses = Int(sqrt(Double(tmp.count)))
        confusionMatrix = Array(repeating: Array(repeating: 0, count: numClasses), count: numClasses)
        for i in 0..<numClasses {
            for j in 0..<numClasses {
                confusionMatrix[i][j] = tmp[i*numClasses+j]
            }
        }
        if let t : TreeNode = try? decoder.decode("tree") {
            tree = t
        }
        else {
            tree = TreeNode()
        }
    }
    
    public init() {
        confusionMatrix = []
        tree = TreeNode()
    }
    
    public func saveToFile(filename : String) {
        do {
            try toJSONString().write(toFile: filename, atomically: false, encoding: .utf8)
        } catch {
            print("Error writing tree to file")
        }
    }
    
    public func toJSON() throws -> Any {
        return try JSONEncoder.create { (encoder) -> Void in
            try encoder.encode(tree, key: "tree")
            try encoder.encode(confusionMatrix.reduce([], {$0 + $1}), key: "confusionMatrix")
        }
    }
    
    public init(filename : String) throws {
        guard let json = String(contentsOfFile: filename, quiet: true) else {
            self.init()
            return
        }
        try self.init(JSONString: json)
    }
    
    public var confusionMatrix : [[Int]] //[predicted][actual]
    public var tree : TreeNode
    
    public func accuracy() -> Double {
        var numCorrect = 0
        for i in 0..<confusionMatrix.count {
            numCorrect += confusionMatrix[i][i]
        }
        return Double(numCorrect)/Double(numTestCases())
    }
    
    public func numTestCases() -> Int {
        var n = 0
        for i in 0..<confusionMatrix.count {
            for j in 0..<confusionMatrix.count {
                n += confusionMatrix[i][j]
            }
        }
        return n
    }
    
    public func macroPrecision() -> Double {
        var precision = 0.0
        for i in 0..<confusionMatrix.count {
            var fp = 0
            for j in 0..<confusionMatrix.count {
                fp += confusionMatrix[i][j]
            }
            if(fp == 0) {
                precision += 1.0
                continue
            }
            fp -= confusionMatrix[i][i]
            precision += Double(confusionMatrix[i][i])/Double(fp + confusionMatrix[i][i])
        }
        return precision/Double(confusionMatrix.count)
    }
    
    public func macroRecall() -> Double {
        var recall = 0.0
        for i in 0..<confusionMatrix.count {
            var fn = 0
            for j in 0..<confusionMatrix.count {
                fn += confusionMatrix[j][i]
            }
            if(fn == 0) {
                recall += 1.0
                continue
            }
            fn -= confusionMatrix[i][i]
            recall += Double(confusionMatrix[i][i])/Double(fn + confusionMatrix[i][i])
        }
        return recall/Double(confusionMatrix.count)
    }
    
    public func macroFMeasure() -> Double {
        let p = macroPrecision()
        let r = macroRecall()
        return (2*p*r)/(p+r)
    }
    
    public func macroFMeasureV2() -> Double {
        var f1 = 0.0
        for i in 0..<confusionMatrix.count {
            var fp = 0
            var fn = 0
            for j in 0..<confusionMatrix.count {
                fp += confusionMatrix[i][j]
                fn += confusionMatrix[j][i]
            }
            fp -= confusionMatrix[i][i]
            fn -= confusionMatrix[i][i]
            let tp = Double(confusionMatrix[i][i])
            f1 += (2.0*tp)/(2*tp+Double(fp)+Double(fn))
        }
        f1 = f1/Double(confusionMatrix.count)
        return f1
    }
}

public class Attribute {
    public var min : Double?
    public var max : Double?
    public var name : String
    public var nominalValues : [String : Int]?
    
    public init(name : String = "", min : Double? = nil, max : Double? = nil) {
        self.name = name
        self.min = min
        self.max = max
        self.nominalValues = nil
    }
    
    public func getValueFromNominal(nominal : String) -> Int {
        if let dict = nominalValues {
            if let v = dict[nominal] {
                return v
            }
        }
        else {
            nominalValues = Dictionary()
        }
        let v = nominalValues!.count
        nominalValues!.updateValue(v, forKey: nominal)
        return v
    }
    
    public func isNominal() -> Bool {
        return nominalValues != nil
    }
}

public class Distribution {
    public var subsets : [[Double]]
    public var numMissing : Double
    private var cachedTotalWeight : Double?
    private var cachedSubsetWeight : [Double?]
    private var cachedDefaultInfo : Double?
    
    public init() {
        subsets = []
        numMissing = 0.0
        cachedTotalWeight = 0.0
        cachedSubsetWeight = []
        cachedDefaultInfo = 0.0
    }
    
    public init(numSubsets : Int, numClasses : Int) {
        subsets = Array(repeating: Array(repeating: 0, count: numClasses), count: numSubsets)
        numMissing = 0.0
        cachedTotalWeight = 0.0
        cachedSubsetWeight = Array(repeating: 0, count: numSubsets)
        cachedDefaultInfo = 0.0
    }
    
    public func invalidateCache() {
        cachedTotalWeight = nil
        cachedSubsetWeight = Array(repeating: nil, count: subsets.count)
        cachedDefaultInfo = nil
    }
    
    public func weightSubset(i : Int) -> Double {
        if let w = cachedSubsetWeight[i] {
            return w
        }
        cachedSubsetWeight[i] = subsets[i].reduce(0, {$0+$1})
        return cachedSubsetWeight[i]!
    }
        
    public func totalWeight() -> Double {
        if let w = cachedTotalWeight {
            return w
        }
        var weight = 0.0
        for s in 0..<subsets.count {
            weight += weightSubset(i: s)
        }
        weight += numMissing
        cachedTotalWeight = weight
        return weight
    }
    
    public func defaultInfo() -> Double {
        if let i = cachedDefaultInfo {
            return i
        }
        cachedDefaultInfo = info(distribution: self, subset: nil)
        return cachedDefaultInfo!
    }
    
}

public class DataSet {
    public var instances : [Point] = []
    public var weights : [Double?] = []
    public var classes : [(value : Int, name : String)]  = []
    public var className : String = ""
    public var attributes : [Attribute] = []
    public var file : String = ""
    
    public init() {
        instances = []
        weights = []
        classes = []
        className = ""
        attributes = []
        file = ""
    }
    
    public init(dataset : DataSet, copyInstances : Bool = true) {
        attributes = dataset.attributes
        if(copyInstances) {
            instances = dataset.instances
            weights = dataset.weights
        }
        else {
            instances = []
            weights = []
            attributes = attributes.map({Attribute(name : $0.name)})
        }
        classes = dataset.classes
        className =  dataset.className
        file = dataset.file
    }
    
    public func sumOfWeights() -> Double {
        var result = 0.0
        for w in weights {
            if(w != nil) {
                result += w!
            }
            else {
                result += 1.0
            }
        }
        return result
    }
    
    public func sortOnAttribute(attribute a: Int) -> Int{
        var numMissing = 0
        instances.sort(by: {(p1 : Point, p2 : Point) -> Bool in
            if(p1.values[a] == nil) {
                return false
            }
            if(p2.values[a] == nil) {
                return true
            }
            return p1.values[a]! < p2.values[a]!
        })
        for i in (0..<instances.count).reversed() {
            if(instances[i].values[a] == nil) {
                numMissing += 1
            }
            else {
                break
            }
        }
        return numMissing
    }
    
    public func addAttribute(name : String) {
        attributes.append(Attribute(name: name))
    }
    
    public func getAttributes() -> [Attribute] {
        return attributes
    }
    
    public func setAttributes(att : [Attribute]) {
        attributes = att
    }
    
    public func getClasses() -> [(value : Int, name : String)] {
        return classes
    }
    
    public func setClasses(classes : [(value : Int, name : String)]) {
        self.classes = classes
    }
    
    public func clearAttributeMinMax() {
        for i in 0..<attributes.count {
            attributes[i].min = nil
            attributes[i].max = nil
        }
    }
    
    public func setClassName(name : String) {
        className = name
    }
    
    public func getClassValue(name : String) -> Int{
        if let i = classes.index(where: {$0.name == name}) {
            return classes[i].value
        }
        else {
            if let v = Int(name) {
                classes.append((name : name ,value: v))
                return v
            }
            else {
                classes.append((name : name, value : classes.count))
                return classes.count - 1
            }
        }
    }
    
    public func getClassIndex(value : Int) -> Int? {
        if let index = classes.index(where: {$0.value == value}) {
            return index
        }
        else {
            return nil
        }
    }
    
    public func addPoint(point : Point, weight: Double? = nil) {
        if(point.values.count == numAttributes()) {
            instances.append(point)
            weights.append(weight)
            for i in 0..<point.values.count {
                if(point.values[i] != nil) {
                    if(attributes[i].min == nil || attributes[i].min! > point.values[i]!) {
                        attributes[i].min = point.values[i]
                    }
                    if(attributes[i].max == nil || attributes[i].max! < point.values[i]!) {
                        attributes[i].max = point.values[i]
                    }
                }
            }
            if let ci = getClassIndex(value: point.classVal) {
                point.classIndex = ci
            }
            else {
                classes.append((value: point.classVal, name: "\(point.classVal)"))
                point.classIndex = classes.count - 1
            }
        }
    }
    
    public func updateAttributeRangeMin(value : Double?, axis : Int) {
        attributes[axis].min = value
    }
    
    public func updateAttributeRangeMax(value : Double?, axis : Int) {
        attributes[axis].max = value
    }
    
    public func numAttributes() -> Int {
        return attributes.count
    }
    
    public func saveAsCSV(file: String) {
        var s = ""
        for i in instances {
            for v in i.values {
                s += "\(v != nil ? String(format: "%.2f", v!) : "?"),"
            }
            s += "\(i.classVal)\n"
        }
        do {
            try s.write(toFile: file, atomically: false, encoding: .utf8)
        }
        catch {
            print("Error writing file")
        }
    }
    
    public func saveAsARFF(file: String) {
        var s = "@relation training\n\n"
        for a in 0..<attributes.count {
            if(attributes[a].name.isEmpty) {
                s += "@attribute att\(a+1) numeric\n"
            }
            else {
                s += "@attribute \(attributes[a].name) numeric\n"
            }
        }
        s += "\n@attribute class {"
        for c in classes {
            s += "\(c.name),"
        }
        s.removeLast()
        s += "}\n\n@data\n"
        for i in instances {
            for v in i.values {
                s += "\(v != nil ? String(format: "%.2f", v!) : "?"),"
            }
            s += "\(classes[i.classIndex].name)\n"
        }
        do {
            try s.write(toFile: file, atomically: false, encoding: .utf8)
        }
        catch {
            print("Error writing file")
        }
    }
    
    public func averages() -> [Double] {
        var averages : [Double] = []
        for d in 0..<attributes.count {
            var sum : Double = 0.0
            var count = 0
            for i in instances {
                if let v = i.values[d] {
                    sum += v
                    count += 1
                }
            }
            averages.append(sum/Double(count))
        }
        return averages
    }
    
    public func fillMissingWith(values : [Double]) {
        for i in instances {
            for a in 0..<i.values.count {
                if(i.values[a] == nil) {
                    i.values[a] = values[a]
                }
            }
        }
    }
    
    public var description : String {get{return instances.description}}
}

public class Point : CustomStringConvertible {
    public var values : [Double?]
    public var classVal : Int
    public var classIndex : Int
    public init(values : [Double?], classVal : Int) {
        self.values = values
        self.classVal = classVal
        self.classIndex = 0
    }
    
    public var description : String {get{return values.description + "-" + String(classVal)}}
}

public func loadFromFile(file: String, sep: Character = ",", headingsPresent : Bool = true, lastIsClass: Bool = true) -> DataSet? {
    let data = DataSet()
    var sep = sep
    guard let f = try? NSString(contentsOfFile: file, encoding: String.Encoding.utf8.rawValue).components(separatedBy: .newlines) else {
        print("Could not open file")
        return nil
    }
    
    if(file.hasSuffix("csv")) {
        sep = ","
    }
    else if(f[0].contains(",")) {
        sep = ","
    }
    
    data.file = file
    
    let l: [Substring] = f[0].split(separator: sep, omittingEmptySubsequences : false)
    let numValues = l.count
    let classVal = lastIsClass ? Int(String(l.last!)) : 0
    for x in (lastIsClass ? l.dropLast() : ArraySlice<Substring>(l)) {
        if(headingsPresent) {
            data.addAttribute(name: String(x))
        }
        else {
            data.addAttribute(name: "")
        }
    }
    
    if(lastIsClass && headingsPresent) {
        data.setClassName(name: String(l.last!))
    }
    
    for l in (headingsPresent ? Array(f.dropFirst()) : f) {
        let s = l.split(separator: sep, omittingEmptySubsequences : false)
        if(s.count != numValues) { //skip line
            continue
        }
        var values: [Double?] = []
        let classVal = lastIsClass ? String(s.last!) : ""
        for i in 0 ..< s.count - (lastIsClass ? 1 : 0) {
            if let v = Double(String(s[i])), !data.attributes[i].isNominal() {
                values += [v]
            }
            else if(s[i] == "" || s[i] == "?") {
                values.append(nil)
            }
            else {
                values += [Double(data.attributes[i].getValueFromNominal(nominal: String(s[i])))]
            }
        }
        
        data.addPoint(point: Point(values: values, classVal: data.getClassValue(name: classVal)))
    }
    
    return data
}

public class AxisSelectionRule : JSONCodable {
	public var rangeMin : Double?
	public var rangeMax : Double?
	public var axisIndex : Int
	public init(rangeMin : Double?, rangeMax : Double?, axisIndex : Int) {
		self.rangeMin = rangeMin
		self.rangeMax = rangeMax
		self.axisIndex = axisIndex
	}
	
	public required init(object: JSONObject) throws {
		let decoder = JSONDecoder(object: object)
		rangeMin = try decoder.decode("rangeMin")
		rangeMax = try decoder.decode("rangeMax")
		axisIndex = try decoder.decode("axisIndex")
	}
}

public class HyperPlaneRule : JSONCodable {
    public required init(object: JSONObject) throws {
        let decoder = JSONDecoder(object: object)
        coefficients = try decoder.decode("coefficients")
    }
    
    public init(coefficients : [Double]) {
        self.coefficients = coefficients
    }
    
    public var coefficients : [Double]
    
}

public typealias PathToNode = [(rule : Rule, invert : Bool)]

public protocol Shape : JSONCodable {
}

public class Rectangle : Shape {
	public var left : Double
	public var right : Double
	public var top : Double
	public var bottom : Double
	
	public init(left : Double, right : Double, top : Double, bottom : Double) {
		self.left = left
		self.right = right
		self.top = top
		self.bottom = bottom
	}
    
    public init() {
        left = 0
        right = 0
        top = 0
        bottom = 0
    }
    
    public init(r : Rectangle) {
        self.left = r.left
        self.right = r.right
        self.top = r.top
        self.bottom = r.bottom
    }
	
	public required init(object: JSONObject) throws {
		let decoder = JSONDecoder(object: object)
		left = try decoder.decode("left")
		right = try decoder.decode("right")
		top = try decoder.decode("top")
		bottom = try decoder.decode("bottom")
	}
}

public class Circle : Shape {
    public required init(object: JSONObject) throws {
        let decoder = JSONDecoder(object: object)
        center.x = try decoder.decode("centerX")
        center.y = try decoder.decode("centerY")
        radius = try decoder.decode("radius")
    }
    
    public var center : (x : Double, y : Double)
    public var radius : Double
    
    public init(center : (x : Double, y : Double), radius : Double) {
        self.center = center
        self.radius = radius
    }
    
    public init(circle : Circle) {
        self.center = circle.center
        self.radius = circle.radius
    }
    
    public func toJSON() throws -> Any {
        return try JSONEncoder.create { (encoder) -> Void in
            try encoder.encode(center.x, key: "centerX")
            try encoder.encode(center.y, key: "centerY")
            try encoder.encode(radius, key: "radius")
        }
    }
    
}

public class RegionRule : JSONCodable {
	public var Attributes : [Int] = []
	public var Region : Shape
	
	public init() {
		Region = Rectangle(left : 0, right : 0, top : 0, bottom : 0)
	}
	
	public required init(object map: JSONObject) throws {
		let decoder = JSONDecoder(object: map)
		Attributes = try decoder.decode("Attributes")
        if let rec = try? decoder.decode("Region") as Rectangle {
            Region = rec
        }
        else {
            Region = try decoder.decode("Region") as Circle
        }
	}
	
    public init(region : Shape, attributes : [Int]) {
		self.Region = region
		self.Attributes = attributes
	}
}

public class PCRegionRule : RegionRule {
	public var axisSeperation : Double
	public var axisMin : [Double]
	public var axisMax : [Double]
    public var AttributeFlipped :  [Bool] = []
	
	public required init(object map: JSONObject) throws {
		let decoder = JSONDecoder(object: map)
		axisSeperation = try decoder.decode("axisSeperation")
		axisMin = try decoder.decode("axisMin")
		axisMax = try decoder.decode("axisMax")
        AttributeFlipped = try decoder.decode("AttributeFlipped")
		try super.init(object: map)
	}
	
	public init(region : Shape, attributes : [Int], axisSeperation : Double, axisMin : [Double], axisMax : [Double], attributesFlipped : [Bool]) {
		self.axisSeperation = axisSeperation
		self.axisMin = axisMin
		self.axisMax = axisMax
        self.AttributeFlipped = attributesFlipped
		super.init(region: region, attributes: attributes)
	}
	
	public override init() {
		axisSeperation = 0
		axisMin = []
		axisMax = []
		super.init()
	}
}

public enum Rule {
	case AxisSelection([AxisSelectionRule])
	case Region([RegionRule])
	case PCRegion([PCRegionRule])
    case HyperPlane(HyperPlaneRule)
}

public class TreeNode : JSONCodable {
	public var parent : TreeNode?
	public var insideChildRule : TreeNode?
	public var outsideChildRule : TreeNode?
	public var classVal : Int?
	public var rules : Rule?
    public var ruleGenerated : Bool
	public var creationTime : String?
	
	public func isLeaf() -> Bool{
		return classVal != nil && !hasChildren()
	}
	
	public func hasChildren() -> Bool {
		return insideChildRule != nil || outsideChildRule != nil
	}
    
    public func description() -> String {
        if(isLeaf()) {
            return ""
        }
        switch rules! {
        case let .AxisSelection(selection):
            var range = ""
            var str  = ""
            if let min = selection.first!.rangeMin {
                range += String(min) + " <= "
            }
            range += String(selection.first!.axisIndex)
            if let max = selection.first!.rangeMax {
                range += " >= " + String(max)
            }
            if(insideChildRule!.isLeaf()) {
                str = range + " : " + String(insideChildRule!.classVal!)
            }
            else {
                str = range + "\n|\t" + insideChildRule!.description().replacingOccurrences(of: "\n", with: "\n|\t")
            }
            if(outsideChildRule!.isLeaf()) {
                str += "\n!(" + range + ") : " + String(outsideChildRule!.classVal!)
            }
            else {
                str += "\n!(" + range + ")" + "\n|\t" + outsideChildRule!.description().replacingOccurrences(of: "\n", with: "\n|\t")
            }
            return str
        default:
            return ""
        }
    }
    
    public func deepestLeaf() -> Int {
        return max(insideChildRule?.deepestLeaf() ?? 0, outsideChildRule?.deepestLeaf() ?? 0)+1
    }
    
    public func sizeOfTree() -> Int {
            return (insideChildRule?.sizeOfTree() ?? 0) +
                (outsideChildRule?.sizeOfTree() ?? 0) + 1
    }
	
	public required init(object: JSONObject) throws {
		let decoder = JSONDecoder(object: object)
		parent = nil
        if let g : Bool = try? decoder.decode("ruleGenerated") {
            ruleGenerated = g
        }
        else {
            ruleGenerated = false
        }
		insideChildRule = try? decoder.decode("insideChildRule")
		outsideChildRule = try? decoder.decode("outsideChildRule")
		insideChildRule?.parent = self
		outsideChildRule?.parent = self
		classVal = try? decoder.decode("classVal")
		creationTime = try? decoder.decode("creationTime")
		if let x : [AxisSelectionRule] = try? decoder.decode("rules") {
			rules = Rule.AxisSelection(x)
		}
		else if let x : [PCRegionRule] = try? decoder.decode("rules") {
			rules = Rule.PCRegion(x)
		}
		else if let x : [RegionRule] = try? decoder.decode("rules") {
			rules = Rule.Region(x)
		}
		else {
			rules = nil
		}
	}
	
	public init() {
		parent = nil
		insideChildRule = nil
		outsideChildRule = nil
		classVal = nil
		rules = nil
        ruleGenerated = false
		let currentDateTime = Date()
		let formatter = DateFormatter()
		formatter.dateStyle = .short
		formatter.timeStyle = .full
		creationTime = formatter.string(from: currentDateTime)
	}
	
	enum TreeNodeError : Error {
		case fileReadError
	}
	
	public convenience init(filename : String) throws {
		guard let json = String(contentsOfFile: filename, quiet: true) else {
			throw TreeNodeError.fileReadError
		}
		try self.init(JSONString: json)
	}
	
	public func toJSON() throws -> Any {
		return try JSONEncoder.create { (encoder) -> Void in
			try encoder.encode(insideChildRule, key: "insideChildRule")
			try encoder.encode(outsideChildRule, key: "outsideChildRule")
			try encoder.encode(classVal, key: "classVal")
			try encoder.encode(creationTime, key: "creationTime")
			if let r = rules {
				switch r {
				case let .AxisSelection(selection):
					try encoder.encode(selection, key: "rules")
				case let .PCRegion(region):
					try encoder.encode(region, key: "rules")
				case let .Region(region):
					try encoder.encode(region, key: "rules")
                case let .HyperPlane(hyperplane):
                    try encoder.encode(hyperplane, key: "rules")
				}
			}
			else {
				try encoder.encode(nil, key: "rules")
			}
		}
	}
	
	public func saveToFile(filename : String) {
		do {
			try toJSONString().write(toFile: filename, atomically: false, encoding: .utf8)
		} catch {
			print("Error writing tree to file")
		}
	}
}

//func insideRules(indexs : Set<Int>, data: DataSet, rules : [Rule]) -> Set<Int> {
//    return insideRules(indexs: indexs, data : data, rules: rules.map({($0, false)}))
//}

public func insideRule(instance : Point, data : DataSet, rule : Rule) -> Bool? {
	switch rule {
	case let .AxisSelection(selection):
		for a in selection {
            guard let v = instance.values[a.axisIndex] else {
                return nil
            }
            if(a.rangeMin != nil && v < a.rangeMin!) {
                return false
            }
            if(a.rangeMax != nil && v > a.rangeMax!) {
                return false
            }
		}
	case let .PCRegion(region):
		for x in region {
            guard let i = insideRegionRule(instance: instance, data: data, rule: x) else {
                return nil
            }
			if(!i) {
				return false
			}
		}
	case let .Region(region):
		for x in region {
            guard let i = insideRegionRule(instance: instance, data: data, rule: x) else {
                return nil
            }
			if(!i) {
				return false
			}
		}
    case let .HyperPlane(hyperplane):
        var value = hyperplane.coefficients[data.numAttributes()]
        for i in 0..<data.numAttributes() {
            value += hyperplane.coefficients[i]*instance.values[i]!
        }
        return value < 0
	}
	return true
}

//func insideRules(indexs : Set<Int>, data : DataSet, rules : PathToNode) -> Set<Int> {
//    var result : Set<Int> = []
//    for i in indexs {
//        var add = true
//        for r in rules {
//            if(insideRule(instance : data.instances[i], data: data, rule: r.rule) == r.invert) {
//                add = false
//                break
//            }
//        }
//        if(add) {
//            result.insert(i)
//        }
//    }
//    return result
//}

public func insideRules(data : DataSet, rules : PathToNode) -> DataSet {
    let result = DataSet(dataset: data, copyInstances: false)
    var add = Array(repeating: true, count: data.instances.count)
    var weights = data.weights
    for r in rules {
        var numIncluded = 0.0
        var numExcluded = 0.0
        var withMissingValue : [Int] = []
        for i in 0..<data.instances.count {
            if(add[i] == false) {
                continue
            }
            if let v = insideRule(instance: data.instances[i], data: data, rule: r.rule) {
                if(v == r.invert) {
                    add[i] = false
                    numExcluded += (weights[i] ?? 1.0)
                }
                else {
                    numIncluded += (weights[i] ?? 1.0)
                }
            }
            else {
                withMissingValue.append(i)
            }
        }
        for i in withMissingValue {
            weights[i] =  (weights[i] ?? 1.0) * (numIncluded/(numIncluded+numExcluded))
        }
    }
    for i in 0..<data.instances.count {
        if(add[i]) {
            result.addPoint(point: data.instances[i], weight: weights[i])
        }
    }
    return result
}

//Taken from https://gist.github.com/ChickenProp/3194723
public func lineRectangleIntersection (line : (x0 : Double, y0 : Double, x1 : Double, y1 : Double),
                   rectangle : (left : Double, right : Double, top : Double, bottom : Double)) -> Bool {
	let vx = line.x1 - line.x0
	let vy = line.y1 - line.y0
	var p = [-vx, vx, -vy, vy]
	var q = [line.x0 - rectangle.left, rectangle.right - line.x0, line.y0 - rectangle.bottom, rectangle.top - line.y0]
	var u1 : Double? = nil
	var u2 : Double? = nil
	
	for i in 0..<4 {
		if (p[i] == 0) {
			if (q[i] < 0) {
				return false
			}
		}
		else {
			let t = q[i] / p[i]
			if (u1 == nil || (p[i] < 0 && u1! < t)) {
				u1 = t
			}
			else if (u2 == nil || (p[i] > 0 && u2! > t)) {
				u2 = t
			}
		}
	}
	
	if (u1! > u2! || u1! > 1 || u1! < 0) {
		return false
	}
	
	//collision.x = x + u1*vx;
	//collision.y = y + u1*vy;
	
	return true;
}

public func lineLineIntersection(p0_x : Double, p0_y : Double,p1_x : Double,p1_y : Double,
                          p2_x : Double, p2_y : Double, p3_x : Double, p3_y : Double) -> (x: Double, y: Double)? {
    let s1_x = p1_x - p0_x
    let s1_y = p1_y - p0_y
    let s2_x = p3_x - p2_x
    let s2_y = p3_y - p2_y
    
    let det = (-s2_x * s1_y + s1_x * s2_y)
    if(abs(det) < 0.0001) {
        return nil
    }
    let s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / det;
    let t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / det;
    
    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
    {
        // Collision detected
        return (x: p0_x + (t * s1_x), y: p0_y + (t * s1_y))
    }
    
    return nil; // No collision
}

public func lineCircleIntersection( line : (x0 : Double, y0 : Double, x1 : Double, y1 : Double), circle : Circle) -> Bool {
    let dx = line.x1 - line.x0
    let dy = line.y1 - line.y0
    let A = dx*dx + dy*dy
    let B = 2 * (dx * (line.x0 - circle.center.x) + dy * (line.y0 - circle.center.y))
    let C = (line.x0 - circle.center.x) * (line.x0 - circle.center.x) + (line.y0 - circle.center.y) * (line.y0 - circle.center.y) - circle.radius*circle.radius
    let det = B*B - 4*A*C
    return A > 0.00000001 && det > 0
}

public func insideRegionRule(instance : Point, data : DataSet, rule : RegionRule) -> Bool? {
    guard let v1 = instance.values[rule.Attributes[0]] else {
        return nil
    }
    guard let v2 = instance.values[rule.Attributes[1]] else {
        return nil
    }
	if let pcrule = rule as? PCRegionRule {
		var start : (x : Double, y : Double) = (0, (v1-pcrule.axisMin[0])/(pcrule.axisMax[0]-pcrule.axisMin[0]))
		var end : (x : Double, y : Double) = (1, (v2-pcrule.axisMin[1])/(pcrule.axisMax[1]-pcrule.axisMin[1]))
        if(pcrule.AttributeFlipped[0]) {
            start.y  = 1 - start.y
        }
        if(pcrule.AttributeFlipped[1]) {
            end.y  = 1 - end.y
        }
        if let rec = rule.Region as? Rectangle {
            return lineRectangleIntersection(line: (start.x, start.y, end.x, end.y), rectangle: (rec.left, rec.right, rec.top, rec.bottom))
        }
        else if let cir = rule.Region as? Circle {
            start.y = 1.7*start.y
            end.x = 0.5
            end.y = 1.7*end.y
            return lineCircleIntersection(line: (start.x, start.y, end.x, end.y), circle: cir)
        }
	}
	else {
        if let rec = rule.Region as? Rectangle {
            return v1 >= rec.left && v1 <= rec.right && v2 >= rec.bottom && v2 <= rec.top
        }
        else if let cir = rule.Region as? Circle {
            let dx = pow(v1-cir.center.x, 2.0)
            let dy = pow(v2-cir.center.y, 2.0)
            return sqrt(dx+dy) < cir.radius
        }
	}
	return false
}

//func insideRegionRule(indexs : Set<Int>, data : DataSet, rule : RegionRule) -> Set<Int> {
//    var result : Set<Int> = []
//    for i in indexs {
//        if insideRegionRule(instance: data.instances[i], data: data, rule: rule) {
//                result.insert(i)
//        }
//    }
//    return result
//}

public func distribution(indexs : Set<Int>, data : [Point]) -> [Int : Int] {
	var maxs : [Int : Int] = [:]
	for i in indexs {
		let c = data[i].classVal
		if let a = maxs[c] {
			maxs[c] = (a + 1)
		}
		else {
			maxs[c] = 1
		}
	}
	return maxs
}

public func distribution(data : DataSet) ->  [Int : Double] {
    var maxs : [Int : Double] = [:]
    for i in 0..<data.instances.count {
        let c = data.instances[i].classVal
        if let a = maxs[c] {
            maxs[c] = (a + (data.weights[i] ?? 1.0))
        }
        else {
            maxs[c] = data.weights[i] ?? 1.0
        }
    }
    return maxs
}

//func mostFreq(indexs : Set<Int>, data : [Point]) -> Int {
//    let maxs = distribution(indexs: indexs, data: data)
//    var max = 0
//    var maxClass = 0
//    for(classVal, count) in maxs {
//        if(count > max) {
//            max = count
//            maxClass = classVal
//        }
//    }
//    return maxClass
//}

public func mostFreq(data : DataSet) -> (Int, [Int : Double]) {
    let maxs = distribution(data: data)
    var max = 0.0
    var maxClass = 0
    for(classVal, count) in maxs {
        if(count > max) {
            max = count
            maxClass = classVal
        }
    }
    return (maxClass, maxs)
}

public func freq(indexs : Set<Int>, data : [Point], c : Int) -> Int {
	var f = 0;
	for i in indexs {
		if(data[i].classVal == c) {
			f += 1;
		}
	}
	return f
}

public func freq(firstIndex : Int, lastIndex : Int, insideRange : Bool, numMissing : Int,  data : [Point], weights: [Double?], c : Int?) -> Double {
    var f = 0.0
    if(insideRange) {
        for i in firstIndex...lastIndex {
            if(c == nil || data[i].classVal == c!) {
                f += weights[i] ?? 1.0
            }
        }
    }
    else {
        if(firstIndex == 0) {
            for i in (lastIndex+1)..<data.count - numMissing {
                if(c == nil || data[i].classVal == c!) {
                    f += weights[i] ?? 1.0
                }
            }
        }
        else {
            for i in 0..<firstIndex {
                if(c == nil || data[i].classVal == c!) {
                    f += weights[i] ?? 1.0
                }
            }
            if(lastIndex < data.count-1) {
                for i in lastIndex+1..<(data.count - numMissing) {
                    if(c == nil || data[i].classVal == c!) {
                        f += weights[i] ?? 1.0
                    }
                }
            }
        }
    }
    return f
}

public func computeFreqTable(data : DataSet) -> [[Double]] {
    var table = Array(repeating: Array(repeating: 0.0, count: data.instances.count), count: data.classes.count+1)
    table[data.getClassIndex(value: data.instances[0].classVal)!][0] = data.weights[0] ?? 1.0
    table[data.classes.count][0] = data.weights[0] ?? 1.0
    for d in 1..<data.instances.count {
        var w = 1.0
        if(data.weights[d] != nil) {
            w = data.weights[d]!
        }
        for c in 0..<data.classes.count {
            if(data.instances[d].classVal == data.classes[c].value) {
                table[c][d] = table[c][d-1] + w
            }
            else {
                table[c][d] = table[c][d-1]
            }
        }
        table[data.classes.count][d] = table[data.classes.count][d-1] + w
    }
    return table
}

public func info(distribution : Distribution, subset : Int?) -> Double {
    var info = 0.0
    if let s = subset {
        for c in distribution.subsets[s] {
            let d = c/distribution.weightSubset(i: s)
            if(!d.isZero) {
                info += d*log2(d)
            }
        }
    }
    else {
        if(distribution.subsets.count == 0) {
            return 0
        }
        for c in 0..<distribution.subsets[0].count {
            var f = 0.0
            for s in 0..<distribution.subsets.count {
                f += distribution.subsets[s][c]
            }
            let d = f/(distribution.totalWeight() - distribution.numMissing)
            if(!d.isZero) {
                info += d*log2(d)
            }
        }
    }
    return -info
}

public func info(firstIndex : Int, lastIndex : Int, insideRange : Bool, numMissing: Int, data : DataSet, freqTable : [[Double]]? = nil) -> Double {
	var info : Double = 0
    var numInstances = 0.0
    if let table = freqTable {
        if(firstIndex == 0) {
            numInstances = table[data.classes.count][lastIndex]
        }
        else {
            numInstances = table[data.classes.count][lastIndex] - table[data.classes.count][firstIndex-1]
        }
        if(!insideRange) {
            numInstances = table[data.classes.count][data.instances.count-1-numMissing] - numInstances
        }
    }
    else {
        numInstances = freq(firstIndex: firstIndex, lastIndex: lastIndex, insideRange: insideRange, numMissing: numMissing, data: data.instances, weights: data.weights, c: nil)
    }
    
	for c in 0..<data.classes.count {
        var f = 0.0
        if let table = freqTable {
            if(firstIndex == 0) {
                f = table[c][lastIndex]
            }
            else {
                f = table[c][lastIndex] - table[c][firstIndex-1]
            }
            if(!insideRange) {
                f = table[c][data.instances.count-1-numMissing] - f
            }
        }
        else {
            f = freq(firstIndex: firstIndex, lastIndex: lastIndex, insideRange: insideRange, numMissing: numMissing, data: data.instances, weights: data.weights, c: data.classes[c].value)
        }
        
		if(f>0) {
			let d = Double(f)/numInstances
			info += d*log2(d)
		}
	}
	return -info
}

public func splitDistribution(firstIndex: Int, lastIndex: Int, numMissing : Int, data : DataSet, freqTable: [[Double]]?) -> (insideCount : Double, outsideCount : Double, missingCount : Double) {
    var insideCount = 0.0
    var outsideCount = 0.0
    var missingCount = 0.0
    if let table = freqTable {
        if(firstIndex == 0) {
            insideCount = table[data.classes.count][lastIndex]
        }
        else {
            insideCount = table[data.classes.count][lastIndex] - table[data.classes.count][firstIndex-1]
        }
        outsideCount = table[data.classes.count][data.instances.count-1-numMissing] - insideCount
        missingCount = table[data.classes.count][data.instances.count-1] - (insideCount+outsideCount)
    }
    else {
        insideCount = freq(firstIndex: firstIndex, lastIndex: lastIndex, insideRange: true, numMissing: numMissing, data: data.instances, weights: data.weights, c: nil)
        outsideCount = freq(firstIndex: firstIndex, lastIndex: lastIndex, insideRange: false, numMissing: numMissing, data: data.instances, weights: data.weights, c: nil)
        missingCount = freq(firstIndex: 0, lastIndex: data.instances.count-1, insideRange: true, numMissing: 0, data: data.instances, weights: data.weights, c: nil) - (insideCount+outsideCount)
    }
    return(insideCount: insideCount, outsideCount: outsideCount, missingCount: missingCount)
}

public func gain(distribution : Distribution) -> Double {
    let t1 = distribution.totalWeight() - distribution.numMissing
    var infox = 0.0
    for s in 0..<distribution.subsets.count {
        infox += (distribution.weightSubset(i: s)/t1)*info(distribution : distribution, subset: s)
    }
    return ((distribution.totalWeight()-distribution.numMissing)/distribution.totalWeight())*(distribution.defaultInfo() - infox)
}

public func gain(firstIndex : Int, lastIndex: Int, numMissing : Int, data : DataSet, freqTable : [[Double]]? = nil) -> Double {
    let distribution = splitDistribution(firstIndex: firstIndex, lastIndex: lastIndex, numMissing: numMissing, data: data, freqTable: freqTable)
    let totalCount = distribution.insideCount + distribution.outsideCount + distribution.missingCount
    let t1 = distribution.insideCount + distribution.outsideCount
    let s1 : Double = (distribution.insideCount/t1)*info(firstIndex : firstIndex, lastIndex : lastIndex, insideRange: true, numMissing: numMissing, data: data, freqTable: freqTable)
    let s2 : Double = (distribution.outsideCount/t1)*info(firstIndex : firstIndex, lastIndex : lastIndex, insideRange: false, numMissing: numMissing, data: data, freqTable: freqTable)
	let infox = s1 + s2
    return ((totalCount-distribution.missingCount)/totalCount)*(info(firstIndex : 0, lastIndex : data.instances.count-1-numMissing, insideRange: true, numMissing: numMissing, data: data, freqTable: freqTable) - infox)
}

public func TwoingCriteria(distribution : Distribution) -> Double {
    var result = 0.0
    for c in 0..<distribution.subsets.count {
        result += abs(distribution.subsets[0][c] - distribution.subsets[1][c])
    }
    result = result * result
    result *= (distribution.weightSubset(i: 0)*distribution.weightSubset(i: 1)/(distribution.totalWeight()*distribution.totalWeight()*4))
    return result
}

public func gainRatio(distribution : Distribution) -> Double{
    var splitInfo = 0.0
    for s in 0..<distribution.subsets.count {
        let w = distribution.weightSubset(i: s)/distribution.totalWeight()
        if(w.isZero) {
            continue
        }
        splitInfo += w*log2(w)
    }
    if(!distribution.numMissing.isZero) {
        let mw = distribution.numMissing/distribution.totalWeight()
        splitInfo += mw*log2(mw)
    }
    let g = gain(distribution : distribution)
    return g / -splitInfo
}

public func gainRatio(firstIndex : Int, lastIndex: Int, numMissing: Int, data : DataSet, freqTable : [[Double]]? = nil) -> Double {
    let distribution = splitDistribution(firstIndex: firstIndex, lastIndex: lastIndex, numMissing: numMissing, data: data, freqTable: freqTable)
    let totalCount = distribution.insideCount + distribution.outsideCount + distribution.missingCount
    
	let r1 = (distribution.insideCount/totalCount)
	let r2 = (distribution.outsideCount/totalCount)
    let r3 = (distribution.missingCount/totalCount)
    let splitInfo = r1*log2(r1) + r2*log2(r2) + (numMissing != 0 ? r3*log2(r3) : 0)
    let g = gain(firstIndex : firstIndex, lastIndex: lastIndex, numMissing: numMissing, data: data, freqTable: freqTable)
    let ret = g / -splitInfo
    return ret
}

public func simulateHumanSplit(data : DataSet) -> Rule? {
    var bestRun : Int = 0
    var bestAttribute : Int = 0
    var bestMin : Double = 0, bestMax : Double = 0
    for a in 0..<data.attributes.count {
        let numMissing = data.sortOnAttribute(attribute: a)
        if(numMissing == data.instances.count) {
            continue
        }
        var currentRun : Int = 0
        var currentMin : Double = data.instances.first!.values[a]!
        var currentClass : Int =  data.instances.first!.classVal
        for i in 0..<(data.instances.count-numMissing) {
            if(data.instances[i].classVal == currentClass) {
                currentRun += 1
            }
            else {
                let x = data.instances[i - 1].values[a]! + (data.instances[i].values[a]! - data.instances[i - 1].values[a]!)/2
                if(currentRun > 1 && currentRun > bestRun) {
                    bestRun = currentRun
                    bestAttribute = a
                    bestMin = currentMin
                    bestMax = x
                }
                currentRun = 1
                currentClass = data.instances[i].classVal
                currentMin = x
            }
        }
        if(currentRun > 1 && currentRun > bestRun) {
            bestRun = currentRun
            bestAttribute = a
            bestMin = currentMin
            bestMax = data.instances[data.instances.count-1-numMissing].values[a]!
        }
    }
    if(bestRun > 1) {
        return Rule.AxisSelection([AxisSelectionRule(rangeMin: bestMin, rangeMax: bestMax, axisIndex: bestAttribute)])
    }
    else {
        return nil
    }
}

protocol ParallelCoordinatesSplit {
    var dataset : DataSet { get }
    var currentAttributes : [Int] { get }
    var bestSingleSplits : [[Double]]? { get set }
}

extension ParallelCoordinatesSplit {
    func NumberOfParameters() -> Int {
        return 4
    }
    
    func GetConstraints() -> [(min: Double, max: Double)] {
        let indexConstraint = (min : 0.0, max : Double(dataset.numAttributes()-1))
        let locationXConstraint = (min : 0.0, max : 1.0)
        let locationYConstraint = (min : 0.0, max : 1.0)
        let widthConstraint = (min: 0.01, max : 1.0)
        let heightConstrain = (min : 0.01, max: 1.0)
        let radiusConstraint = (min : 0.01, max : 0.5)
        return [locationXConstraint, locationYConstraint, locationXConstraint, locationYConstraint]
    }
    
    func EvaluateCost(parameters: [Double]) -> Double {
        let att = currentAttributes.map { abs($0) }
        let flipped = currentAttributes.map { $0 < 0}
        //let rule = PCRegionRule(region: Circle(center: (x : parameters[0], y : parameters[1]), radius: parameters[2]), attributes: att, axisSeperation : 0.5, axisMin : [dataset.attributes[att[0]].min!, dataset.attributes[att[1]].min!], axisMax : [dataset.attributes[att[0]].max!, dataset.attributes[att[1]].max!], attributesFlipped : [false,false])
        let rule = PCRegionRule(region: Rectangle(left: parameters[0], right: parameters[2], top: parameters[1], bottom: parameters[3]), attributes: att, axisSeperation: 0.5, axisMin: [dataset.attributes[att[0]].min!, dataset.attributes[att[1]].min!], axisMax: [dataset.attributes[att[0]].max!, dataset.attributes[att[1]].max!], attributesFlipped : flipped)
        let distribution = Distribution(numSubsets: 2, numClasses: dataset.classes.count)
        for p in 0..<dataset.instances.count {
            if let inRule = insideRegionRule(instance: dataset.instances[p], data: dataset, rule: rule) {
                if(inRule) {
                    distribution.subsets[0][dataset.instances[p].classIndex] += dataset.weights[p] ?? 1.0
                }
                else {
                    distribution.subsets[1][dataset.instances[p].classIndex] += dataset.weights[p] ?? 1.0
                }
            }
            else {
                distribution.numMissing += dataset.weights[p] ?? 1.0
            }
        }
        distribution.invalidateCache()
        if(distribution.weightSubset(i: 0).isZero || distribution.weightSubset(i: 1).isZero) {
            return Double.infinity
        }
        let g = gain(distribution: distribution)
        if(g < 0.00001) {
            return Double.infinity
        }
        else {
            return 1/g //+ parameters[2]*parameters[3]
        }
    }
    
    mutating func CalculateBestSingleSplits() {
        var bestSingleSplits : [[Double]] = Array(repeating: [], count: dataset.numAttributes())
        let sema = DispatchSemaphore(value: 1)
        let group = DispatchGroup()
        let d = dataset
        for attribute in 0..<dataset.numAttributes() {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                let data = DataSet(dataset: d, copyInstances: true)
                let a = data.attributes[attribute]
                let (split, _) = findBestSplit(data: data, attribute: attribute)
            
                if let s = split {
                    switch s {
                    case let .AxisSelection(selection):
                        var top = 1.0
                        var bottom = 0.0
                        if let m = selection.first?.rangeMax {
                            top = (m - a.min!)/(a.max! - a.min!)
                        }
                        if let m = selection.first?.rangeMin {
                            bottom = (m - a.min!)/(a.max! - a.min!)
                        }
                        /*if(currentAttributes[attribute] < 0) {
                         top = 1.0 - top
                         bottom = 1.0 - bottom
                         }*/
                        bestSingleSplits[attribute] = [0, top, 0.01, bottom]
                        /*if(attribute == 0) {
                         bestSingleSplits![0] = [0, top, 0.01, bottom]
                         }
                         else {
                         bestSingleSplits![1] = [0.99, top, 1.0, bottom]
                         }*/
                        //bestSingleSplits![attribute] = [Double(attribtue), ]//[Double(attribute), (top+bottom)/2.0, 0.01, abs(top-bottom)]
                    default:
                        print("error")
                    }
                }
                group.leave()
            }
        }
        group.wait()
        self.bestSingleSplits = bestSingleSplits
    }
}

public class Progress {
    init(tasks : Int) {
        numTasks = tasks
        sema = DispatchSemaphore(value: 1)
        startTime = Date().timeIntervalSince1970
        tasksComplete = 0
    }
    
    func reset(tasks: Int) {
        numTasks = tasks
        startTime = Date().timeIntervalSince1970
        tasksComplete = 0
    }
    
    func taskComplete() {
        sema.wait()
        tasksComplete += 1
        sema.signal()
    }
    
    func progress() -> (complete : Double, time : Double) {
        if(numTasks == 0) {
            return (1.0, Date().timeIntervalSince1970-startTime)
        }
        return (Double(tasksComplete)/Double(numTasks), Date().timeIntervalSince1970-startTime)
    }
    
    private var numTasks : Int
    private var tasksComplete : Int
    private var startTime : Double
    private var sema : DispatchSemaphore
}

public struct DifferentialEvoluationSplit : DifferentialEvolution, ParallelCoordinatesSplit {
    mutating func GetInitialCandidate() -> [Double] {
        var result : [Double] = []
        if(Bool.random()) {
            result = bestSingleSplits![abs(currentAttributes[0])]
            if(currentAttributes[0] < 0 && result.count == 4) {
                result[1] = 1.0 - result[1]
                result[3] = 1.0 - result[3]
            }
        }
        else {
            result = bestSingleSplits![abs(currentAttributes[1])]
            if(result.count == 4) {
                result[0] = 0.99
                result[2] = 1.0
                if(currentAttributes[1] < 0) {
                    result[1] = 1.0 - result[1]
                    result[3] = 1.0 - result[3]
                }
            }
        }
        return result
    }
    
    public mutating func getRule(progress: Progress? = nil) -> Rule? {
        var bestAttIndex = [0,1]
        var bestCost = Double.infinity
        var bestParameters : [Double] = []
        var bestIsFlipped = false
        let sema = DispatchSemaphore(value: 1)
        let group = DispatchGroup()
        progress?.reset(tasks: 100*50*dataset.numAttributes()*(dataset.numAttributes()-1))
        CalculateBestSingleSplits()
        var cp = self
        for i in 0..<(dataset.numAttributes()-1) {//dataset.numAttributes() {
            for j in (i+1)...(i+1) {//<dataset.numAttributes() {
                if(dataset.attributes[i].min == nil || dataset.attributes[j].min == nil) { //attriubte value is missing for all instances
                    continue
                }
                for k in [false,true] {
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async { [i,j,k] in
                    var copy = cp
                    copy.currentAttributes = [i, k ? -j : j]
                    copy.dataset = DataSet(dataset: cp.dataset, copyInstances: true)
                    var parameters = copy.Optimize(iterations: 100, populationSize: 50, mutationFactor: 0.6, crossoverFactor: 0.4, percentageInitialProvided: 0.15, onIterationComplete: {progress?.taskComplete()})
                    let c = copy.EvaluateCost(parameters: parameters)
                    sema.wait()
                    if(c < bestCost) {
                        bestCost = c
                        bestAttIndex = [i,j]
                        bestIsFlipped = k
                        bestParameters = parameters
                    }
                    sema.signal()
                    group.leave()
                    }
                }
            }
        }
        
        group.wait()
        
        if(bestCost.isInfinite) {
            return nil
        }
        
        let att = bestAttIndex
        //print("Final Cost \(bestCost)")
        //return Rule.PCRegion([PCRegionRule(region: Circle(center: (x : bestParameters[0], y : bestParameters[1]), radius: bestParameters[2]), attributes: att, axisSeperation : 0.5, axisMin : [dataset.attributes[att[0]].min!, dataset.attributes[att[1]].min!], axisMax : [dataset.attributes[att[0]].max!, dataset.attributes[att[1]].max!], attributesFlipped : [false,false])])
        return Rule.PCRegion([PCRegionRule(region: Rectangle(left: bestParameters[0], right: bestParameters[2], top: bestParameters[1], bottom: bestParameters[3]), attributes: att, axisSeperation: 0.5, axisMin: [dataset.attributes[att[0]].min!, dataset.attributes[att[1]].min!], axisMax: [dataset.attributes[att[0]].max!, dataset.attributes[att[1]].max!], attributesFlipped : [false,bestIsFlipped])])
    }
    
    public init(dataset : DataSet) {
        self.dataset = dataset
        currentAttributes = [0,0]
    }
    
    public var dataset : DataSet
    internal var currentAttributes : [Int]
    internal var bestSingleSplits : [[Double]]?
}

public struct HillClimberSplit: ParallelCoordinatesSplit {
    
    public enum Mode {
        case FirstImprovement
        case BestImprovement
        case RoundRobinImprovement
    }
    public init(dataset : DataSet, mode m : Mode = .FirstImprovement) {
        self.dataset = dataset
        currentAttributes = [0,0]
        mode = m
        intersectionsSortedX = []
        intersectionsSortedY = []
    }
    
    mutating func CalculateIntersections() {
        intersectionsSortedX = []
        intersectionsSortedY = []
        func yVal0(v : Double) -> Double {
            return v*(dataset.attributes[currentAttributes[0]].max!-dataset.attributes[currentAttributes[0]].min!)+dataset.attributes[currentAttributes[0]].min!
        }
        func yVal1(v : Double) -> Double{
            return v*(dataset.attributes[currentAttributes[1]].max!-dataset.attributes[currentAttributes[1]].min!)+dataset.attributes[currentAttributes[1]].min!
        }
        for i in 0..<dataset.instances.count {
            if(dataset.instances[i].values[currentAttributes[0]] == nil ||
                dataset.instances[i].values[currentAttributes[1]] == nil
                ) {
                continue
            }
            let y0 = yVal0(v: dataset.instances[i].values[currentAttributes[0]]!)
            let y1 = yVal1(v: dataset.instances[i].values[currentAttributes[1]]!)
            for j in (i+1)..<dataset.instances.count {
                if(dataset.instances[j].values[currentAttributes[0]] == nil ||
                    dataset.instances[j].values[currentAttributes[1]] == nil
                    ) {
                    continue
                }
                let y2 = yVal0(v: dataset.instances[j].values[currentAttributes[0]]!)
                let y3 = yVal1(v: dataset.instances[j].values[currentAttributes[1]]!)
                if let intersection = lineLineIntersection(p0_x: 0, p0_y: y0, p1_x: 1.0, p1_y: y1, p2_x: 0, p2_y: y2, p3_x: 1.0, p3_y: y3) {
                    if let indexX = intersectionsSortedX.firstIndex(where: {$0.x > intersection.x}) {
                        intersectionsSortedX.insert(intersection, at: indexX)
                    }
                    else {
                        intersectionsSortedX.append(intersection)
                    }
                    if let indexY = intersectionsSortedY.firstIndex(where: {$0.y > intersection.y}) {
                        intersectionsSortedX.insert(intersection, at: indexY)
                    }
                    else {
                        intersectionsSortedY.append(intersection)
                    }
                }
            }
        }
    }
    
    public mutating func getRule() -> Rule? {
        var bestAttIndex = [0,1]
        var bestCost = Double.infinity
        var bestParameters : [Double] = []
        let sema = DispatchSemaphore(value: 1)
        let group = DispatchGroup()
        CalculateBestSingleSplits()
        var cp = self
        for i in 0..<dataset.numAttributes() {
            for j in (i+1)..<dataset.numAttributes() {
                if(dataset.attributes[i].min == nil || dataset.attributes[j].min == nil) { //attriubte value is missing for all instances
                    continue
                }
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async { [i,j] in
                    var copy = cp
                    copy.currentAttributes = [i, j]
                    copy.dataset = DataSet(dataset: cp.dataset, copyInstances: true)
                
                    var currentCandidate = copy.bestSingleSplits![copy.currentAttributes[0]]
                    if(currentCandidate.count == 0) {
                        currentCandidate = [Double.random(in: 0...1), Double.random(in: 0...1), Double.random(in: 0...1), Double.random(in: 0...1)]
                    }
                    else {
                        let x1 = Double.random(in: 0...1)
                        let x2 = Double.random(in: 0...1)
                        currentCandidate[0] = min(x1, x2)
                        currentCandidate[2] = max(x1, x2)
                    }
                    var currentCost = copy.EvaluateCost(parameters: currentCandidate)
                    let maxIterations = 300
                    let stepSize = 0.05
                    var movements : [(Index: Int, Direction: Double)] = []
                    for i in 0..<copy.NumberOfParameters() {
                        for direction in [-1.0, 1.0] {
                            movements.append((Index: i, Direction: direction))
                        }
                    }
                    //CalculateIntersections()
                
                    for _ in 0..<maxIterations {
                        var bestNewCandidate : [Double]? = nil
                        var bestNewCost = currentCost
                        for i in 0..<movements.count {
                            var newCandidate = currentCandidate
                            newCandidate[movements[i].Index] += stepSize*movements[i].Direction
                            let newCost = copy.EvaluateCost(parameters: newCandidate)
                            if(newCost < bestNewCost) {
                                bestNewCandidate = newCandidate
                                bestNewCost = newCost
                            }
                            if(bestNewCandidate != nil && copy.mode != .BestImprovement) {
                                if(copy.mode == .RoundRobinImprovement) {
                                    for _ in (i+1)..<movements.count {
                                        movements.append(movements.popLast()!)
                                    }
                                }
                                break
                            }
                        }
                        if(bestNewCandidate != nil) {
                            currentCandidate = bestNewCandidate!
                            currentCost = bestNewCost
                        }
                        else {
                            break
                        }
                    }
                    sema.wait()
                    if(currentCost < bestCost) {
                        bestCost = currentCost
                        bestAttIndex = [i,j]
                        bestParameters = currentCandidate
                    }
                    sema.signal()
                    group.leave()
                }
            }
        }
        
        group.wait()
        
        if(bestCost.isInfinite) {
            return nil
        }
        
        let att = bestAttIndex
        //print("Final Cost \(bestCost)")
        return Rule.PCRegion([PCRegionRule(region: Rectangle(left: bestParameters[0], right: bestParameters[2], top: bestParameters[1], bottom: bestParameters[3]), attributes: att, axisSeperation: 0.5, axisMin: [dataset.attributes[att[0]].min!, dataset.attributes[att[1]].min!], axisMax: [dataset.attributes[att[0]].max!, dataset.attributes[att[1]].max!], attributesFlipped : [false,false])])
    }
    
    public var dataset : DataSet
    internal var currentAttributes : [Int]
    internal var bestSingleSplits : [[Double]]?
    internal var intersectionsSortedX: [(x: Double, y: Double)]
    internal var intersectionsSortedY: [(x: Double, y: Double)]
    public var mode : Mode
}

public func findBestSplit(data : DataSet, attribute : Int? = nil, twoValueSplit : Bool = false) -> (rule: Rule?, gainRatio: Double) {
    var range = 0..<data.attributes.count
    if let att = attribute {
        range = att..<(att+1)
    }
    var bestGainRatio : [Double?] = Array(repeating: nil, count: range.count)
    var bestGain : [Double?] = Array(repeating: nil, count: range.count)
	var bestAttribute : Int = 0
	var bestMin : [Double?] = Array(repeating: nil, count: range.count), bestMax : [Double?] = Array(repeating: nil, count: range.count)
    var validModels = 0
    var averageGain = 0.0
    //let group = DispatchGroup()
    //let sema = DispatchSemaphore(value: 1)
    var index = 0
	for a in range {
        //group.enter()
        //DispatchQueue.global(qos: .userInitiated).async { [a] in
            var bestI = 0, bestJ = 0
            let dataCopy = DataSet(dataset: data, copyInstances: true)
            let numMissing = dataCopy.sortOnAttribute(attribute: a)
            if(numMissing == dataCopy.instances.count) {
                //group.leave()
                //return
                break
            }
            let freqTable = computeFreqTable(data: dataCopy)
            var minVal : Double = 0
            var maxVal : Double = 0
            let minSplit = min(max(2.0, 0.1*Double(Double(dataCopy.instances.count-numMissing)/Double(dataCopy.classes.count))), 25.0)
            var lastMin : Double? = nil
            var iRange = 0...0
            if(twoValueSplit) {
                iRange = 0...(dataCopy.instances.count-2)
            }
            for i in iRange{
                if(dataCopy.instances[i].values[a] == nil) {
                    break
                }
                minVal = dataCopy.instances[i].values[a]!
                if(lastMin != nil && abs(minVal - lastMin!) < 0.0001) {
                    continue
                }
                lastMin = minVal
                
                for j in (i+1)..<dataCopy.instances.count {
                    if(dataCopy.instances[j].values[a] == nil) {
                        continue
                    }
                    maxVal = dataCopy.instances[j].values[a]!
                    if(j < dataCopy.instances.count-1 && dataCopy.instances[j+1].values[a] != nil && dataCopy.instances[j+1].values[a]! - maxVal < 0.0001) {
                        continue
                    }
                    let g = gain(firstIndex : i , lastIndex: j, numMissing: numMissing, data: dataCopy, freqTable: freqTable)
                    if(bestGain[index] == nil || g > bestGain[index]!) {
                        let distribution = splitDistribution(firstIndex: i, lastIndex: j, numMissing: numMissing, data: dataCopy, freqTable: freqTable)
                        if(distribution.insideCount >= minSplit && distribution.outsideCount >= minSplit) {
                            bestI = i
                            bestJ = j
                            bestGain[index] = g
                        }
                    }
                }
            }
            var gr : Double? = nil
            if(bestGain[index] != nil) {
                gr = gainRatio(firstIndex: bestI, lastIndex: bestJ, numMissing: numMissing, data: dataCopy, freqTable: freqTable)
                validModels += 1
                averageGain += bestGain[index]!
            }
            //sema.wait()
            if(gr != nil) {
                bestGainRatio[index] = gr
                minVal = dataCopy.instances[bestI].values[a]!
                maxVal = dataCopy.instances[bestJ].values[a]!
                //put the split points half way between the next value above/bellow
                if(bestI != 0) {
                    minVal = minVal - (minVal-dataCopy.instances[bestI-1].values[a]!)/2
                }
                if(bestJ < dataCopy.instances.count - 1 && dataCopy.instances[bestJ+1].values[a] != nil) {
                    maxVal = maxVal + (dataCopy.instances[bestJ+1].values[a]!-maxVal)/2
                }
                
                //if the min or max is the datasets min or max just split on one value
                if(minVal > dataCopy.instances[0].values[a]!) {
                    bestMin[index] = minVal
                }
                else {
                    bestMin[index] = nil
                }
                if(maxVal < dataCopy.instances[dataCopy.instances.count-1-numMissing].values[a]!) {
                    bestMax[index] = maxVal
                }
                else {
                    bestMax[index] = nil
                }
            }
            //sema.signal()
            //group.leave()
        //}
        index += 1
	}
    //group.wait()
    
	if(validModels == 0) {
        return (rule: nil, gainRatio: 0.0)
	}
    averageGain /= Double(validModels)
    
    index = 0
    var bestModelGr = 0.0
    var bestIndex = 0
    for a in range {
        if let g = bestGainRatio[index], bestGain[index]! >= (averageGain-0.001) && g > bestModelGr {
            bestModelGr = g
            bestIndex = index
            bestAttribute = a
        }
        index += 1
    }
    
    
    //print("Gain Ratio \(bestGainRatio!)")
    return (rule: Rule.AxisSelection([AxisSelectionRule(rangeMin : bestMin[bestIndex], rangeMax : bestMax[bestIndex], axisIndex : bestAttribute)]), gainRatio: bestModelGr)
}

public func rulesForNode(n : TreeNode) -> PathToNode {
	if(n.parent == nil) {
		return []
	}
	if(n.parent!.insideChildRule! === n) {
		return rulesForNode(n: n.parent!) + [(n.parent!.rules!, false)]
	}
	else {
		return rulesForNode(n: n.parent!) + [(n.parent!.rules! ,true)]
	}
}

func errorFromLargestBranch(node : TreeNode, data : DataSet) -> (error: Double, branch: TreeNode) {
    let inside = insideRules(data: data, rules: rulesForNode(n: node.insideChildRule!))
    let largestBranch = inside.instances.count >= (data.instances.count - inside.instances.count) ? node.insideChildRule! : node.outsideChildRule!
    func branchError(n : TreeNode, data : DataSet) -> Double {
        if(n.isLeaf()) {
            let leafClasses : (classVal : Int, distribution : [Int:Double]) = mostFreq(data: data)
            let leafError = errorEstimate(e: data.sumOfWeights()-leafClasses.distribution[leafClasses.classVal]!, N: data.sumOfWeights())
            return leafError
        }
        let insideData = insideRules(data: data, rules: rulesForNode(n: n.insideChildRule!))
        let outsideData = insideRules(data: data, rules: rulesForNode(n: n.outsideChildRule!))
        return branchError(n: n.insideChildRule!, data: insideData) + branchError(n: n.outsideChildRule!, data: outsideData)
    }
    largestBranch.parent = nil
    let error = branchError(n: largestBranch, data: data)
    largestBranch.parent = node
    return (error, largestBranch)
}

public func pruneNode(node : TreeNode, data : DataSet) -> Double {
    let instances = insideRules(data: data, rules: rulesForNode(n: node))
    let leafClasses : (classVal : Int, distribution : [Int:Double]) = mostFreq(data: instances)
    let leafError = errorEstimate(e: instances.sumOfWeights()-leafClasses.distribution[leafClasses.classVal]!, N: instances.sumOfWeights())
    if(node.hasChildren()) {
        let e1 = pruneNode(node: node.insideChildRule!, data : instances)
        let e2 = pruneNode(node: node.outsideChildRule!, data : instances)
        let treeError = (e1 + e2)
        let branchError = errorFromLargestBranch(node: node, data: instances)
    
        if(leafError < treeError+0.1 && leafError < branchError.error+0.1) {
            node.insideChildRule = nil
            node.outsideChildRule = nil
            node.classVal = leafClasses.classVal
            node.rules = nil
            return leafError
        }
        else {
            if(branchError.error < treeError+0.1) {
                branchError.branch.parent = node.parent
                if(node.parent?.insideChildRule === node) {
                    node.parent?.insideChildRule = branchError.branch
                }
                else {
                    node.parent?.outsideChildRule = branchError.branch
                }
                return pruneNode(node: branchError.branch, data: instances)
            }
            else {
                return e1 + e2
            }
        }
    }
    else {
        return leafError
    }
}

public func errorEstimate(e : Double, N : Double) -> Double {
    let CF = 0.25
    let z = 0.6744897501960816
    if(e < 1) {
        return Double(N)*(1 - pow(CF, 1/Double(N)))
    }
    if(e >= N) {
        return 0
    }
    let f = (e+0.5)/N
    var error = f + (z*z)/(2*N)
    var s = f/N - f*f/N
    s += (z*z)/(4*N*N)
    s = sqrt(s)
    error += z*s
    error =  error/(1+z*z/N)
    return error*N
}


public func classifyPoint(point : Point, dataset : DataSet, classifier : TreeNode) -> Int {
    func classify(classifier: TreeNode) -> [Int : Double] {
        var probabilities : [Int : Double] = [:]
        if(!classifier.hasChildren()) {
            if(!classifier.isLeaf()) {
                classifier.classVal = mostFreq(data: insideRules(data: dataset, rules: rulesForNode(n: classifier))).0
            }
            probabilities[classifier.classVal!] = 1.0
            return probabilities
        }
        if let inside = insideRule(instance: point, data: dataset, rule: classifier.rules!) {
            if(inside) {
                return classify(classifier: classifier.insideChildRule!)
            }
            else {
                return classify(classifier: classifier.outsideChildRule!)
            }
        }
        else {
            let p1 = classify(classifier: classifier.insideChildRule!)
            let p2 = classify(classifier: classifier.outsideChildRule!)
            let inside = insideRules(data: dataset, rules: rulesForNode(n: classifier.insideChildRule!))
            let outside = insideRules(data: dataset, rules: rulesForNode(n: classifier.outsideChildRule!))
            var insideWeight = inside.sumOfWeights()
            var outsideWeight = outside.sumOfWeights()
            let totalWeight = insideWeight + outsideWeight
            insideWeight = insideWeight/totalWeight
            outsideWeight = outsideWeight/totalWeight
            for p in p1 {
                probabilities[p.key] = p.value*insideWeight
            }
            for p in p2 {
                if(probabilities[p.key] == nil) {
                    probabilities[p.key] = p.value*insideWeight
                }
                else {
                    probabilities[p.key] = probabilities[p.key]! + p.value*insideWeight
                }
            }
            return probabilities
        }
    }
    let probabilities = classify(classifier: classifier)
    var max = 0.0
    var classification = 0
    for p in probabilities {
        if(p.value > max) {
            classification = p.key
            max = p.value
        }
    }
    return classification
}

public func findUnfinished(n : TreeNode) -> TreeNode? {
    if(n.isLeaf()) {
        return nil
    }
    if(n.insideChildRule == nil && n.outsideChildRule == nil && n.classVal == nil) {
        return n;
    }
    if let i = findUnfinished(n: n.insideChildRule!) {
        return i
    }
    return findUnfinished(n: n.outsideChildRule!)
}

public func exportAsDecisionTree(filename : String, trainingData: DataSet, classifier: TreeNode) {
    while true {
        if let n = findUnfinished(n: classifier) {
            n.classVal = mostFreq(data: insideRules(data: trainingData, rules: rulesForNode(n: n))).0
        }
        else {
            break
        }
    }
    classifier.saveToFile(filename: filename + (trainingData.file.components(separatedBy: "/").last)!)
}

public func exportAsDecisionList(file : String, precision : Int, trainingData: DataSet, classifier: TreeNode) {
    while true {
        if let n = findUnfinished(n: classifier) {
            n.classVal = mostFreq(data: insideRules(data: trainingData, rules: rulesForNode(n: n))).0
        }
        else {
            break
        }
    }
    if let dl = DecisionList(fromDecisionTree: classifier, dataset: trainingData) {
        dl.simplifyRules()
        dl.saveToFile(file: file, precision: precision)
    }
}

public func testClassifier(points : [Point], dataset : DataSet, classifier : TreeNode) -> Double{
    var correct : Double = 0
    for p in points {
        if(classifyPoint(point: p, dataset : dataset, classifier: classifier) == p.classVal) {
            correct += 1
        }
    }
    //print("Accuracy \(correct/Double(points.count))%")
    return correct/Double(points.count)
}

public enum BuildMethod {
    case C45
    case DE
    case HCF
    case HCB
    case HCR
    case OC1
}

public func buildOC1(data: DataSet) -> TreeNode {
    data.fillMissingWith(values: data.averages())
    var values = data.instances.map {[0.0] + $0.values.map {Float($0!)}}
    var points : [POINT] = []
    for i in 0..<data.instances.count {
        points.append(POINT(dimension: &values[i][0], category: Int32(data.instances[i].classIndex+1), val: 0))
    }
    setDatasetParams(Int32(data.attributes.count), Int32(data.classes.count))
    allocate_structures(Int32(data.instances.count))
    var pp : [UnsafeMutablePointer<point>?]? = []
    pp!.append(UnsafeMutablePointer<point>?.none)
    for i in 0..<points.count {
        pp!.append(&points[i])
    }
    let ptr = UnsafeMutablePointer(mutating: pp)
    var file = "/tmp/tree"
    var filePtr = strdup(file)
    let tree = build_tree(ptr, Int32(data.instances.count), filePtr)
    free(filePtr)
    func CTreeToTree(tree : UnsafeMutablePointer<tree_node>?) -> TreeNode {
        let node = TreeNode()
        if let c = tree?.pointee.coefficients {
            var coefficients : [Double] = []
            for i in 0..<(data.numAttributes()+1) {
                coefficients.append(Double(c[i+1]))
            }
            node.rules = .HyperPlane(HyperPlaneRule(coefficients: coefficients))
            if(tree?.pointee.left != nil) {
                node.insideChildRule = CTreeToTree(tree: tree?.pointee.left)
            }
            else {
                node.insideChildRule = TreeNode()
                node.insideChildRule?.classVal = data.classes[Int(tree!.pointee.left_cat)-1].value
                
            }
            if(tree?.pointee.right != nil) {
                node.outsideChildRule = CTreeToTree(tree: tree?.pointee.right)
            }
            else {
                node.outsideChildRule = TreeNode()
                node.outsideChildRule?.classVal = data.classes[Int(tree!.pointee.right_cat)-1].value
            }
            node.insideChildRule!.parent = node
            node.outsideChildRule!.parent = node
        }
        return node
    }
    let root = CTreeToTree(tree: tree)
    return root
}

public func finishSubTree(node : TreeNode, data : DataSet, fullTrainingSet: DataSet, buildMethod : BuildMethod, progress: Progress? = nil) {
    var insideInstances : DataSet? = nil
    if(!node.hasChildren()) {
        var stop = true
        if(data.instances.count > 2) { //Int(Double(allData.instances.count)*0.05)) {
            for i in data.instances {
                if(i.classVal != data.instances[0].classVal) {
                    stop = false
                    break
                }
            }
        }
        if(!stop) {
            switch buildMethod {
            case .C45:
                node.rules = findBestSplit(data: data).rule
            case .DE:
                var x = DifferentialEvoluationSplit(dataset: data)
                node.rules = x.getRule(progress: progress)
            case .HCF:
                var hc = HillClimberSplit(dataset: data)
                node.rules = hc.getRule()
            case .HCB:
                var hc = HillClimberSplit(dataset: data, mode: .BestImprovement)
                node.rules = hc.getRule()
            case .HCR:
                var hc = HillClimberSplit(dataset: data, mode: .RoundRobinImprovement)
                node.rules = hc.getRule()
            case .OC1:
                let OC1Node = buildOC1(data: data)
                node.rules = OC1Node.rules
                node.insideChildRule = OC1Node.insideChildRule
                node.outsideChildRule = OC1Node.outsideChildRule
                node.classVal = OC1Node.classVal
                node.creationTime = OC1Node.creationTime
                return
            }
            if(node.rules == nil) {
                stop = true
            }
            else {
                node.ruleGenerated = true
                node.insideChildRule = TreeNode()
                node.insideChildRule!.parent = node
                insideInstances = insideRules(data: data, rules: rulesForNode(n: node.insideChildRule!))
                if(insideInstances!.instances.count < 2 || insideInstances!.instances.count > data.instances.count - 2) {
                    stop = true
                    node.rules = nil
                    node.insideChildRule = nil
                }
            }
        }
        if(stop) {
            node.insideChildRule = nil
            node.outsideChildRule = nil
            if(!data.instances.isEmpty) {
                node.classVal = mostFreq(data: insideRules(data: fullTrainingSet, rules: rulesForNode(n: node))).0
            }
            return
        }
        node.insideChildRule = TreeNode()
        node.outsideChildRule = TreeNode()
        node.insideChildRule?.parent = node
        node.outsideChildRule?.parent = node
    }
    
    finishSubTree(node: node.insideChildRule!, data: insideInstances!, fullTrainingSet: fullTrainingSet, buildMethod: buildMethod)
    finishSubTree(node: node.outsideChildRule!, data: insideRules(data: data, rules: rulesForNode(n: node.outsideChildRule!)), fullTrainingSet: fullTrainingSet, buildMethod: buildMethod)
}

public class CrossValidationProgress {
    public init() {
        foldProgress = Progress(tasks: 0)
        numFolds = 0
        completedFolds = 0
        startTime = Date().timeIntervalSince1970
    }
    
    internal func reset(folds : Int) {
        numFolds = folds
        completedFolds = 0
        startTime = Date().timeIntervalSince1970
    }
    
    public func progress() -> (complete : Double, time : Double) {
        if(numFolds == 0) {
            return (1, Date().timeIntervalSince1970-startTime)
        }
        let complete = (Double(completedFolds) + foldProgress.progress().complete)/Double(numFolds)
        return (complete, Date().timeIntervalSince1970-startTime)
    }
    
    internal var numFolds : Int
    internal var completedFolds : Int
    internal var foldProgress : Progress
    private var startTime : Double
}

public func crossValidation(data : DataSet, folds : Int = 10, buildMethod : BuildMethod, progress : CrossValidationProgress? = nil, runParallel: Bool = false) -> [Result] {
    
    if let p = progress {
        p.reset(folds: folds)
    }
    let copy = DataSet(dataset: data, copyInstances: true)
    var foldSets : [DataSet] = []
    for _ in 0..<folds {
        foldSets.append(DataSet(dataset: data, copyInstances: false))
    }
    
    copy.instances.shuffle()
    copy.instances.sort { (p1 : Point, p2 : Point) -> Bool in
        p1.classVal <= p2.classVal
    }
    
    var index = 0
    while(index < copy.instances.count) {
        for i in 0..<folds {
            if(index >= copy.instances.count) {
                break
            }
            //let r = Int.random(in: 0..<copy.instances.count)
            foldSets[i].addPoint(point: copy.instances[index])
            index += 1
        }
    }
    
    
    var result = Array(repeating: Result(), count: folds)
    let group = DispatchGroup()
    for  i in 0..<folds {
        func runFold(fold : Int) {
            result[fold].tree = TreeNode()
            let d = DataSet(dataset: data, copyInstances : false)
            for j in 0..<folds {
                if(j == fold) {
                    continue
                }
                for instance in foldSets[j].instances {
                    d.addPoint(point: instance)
                }
            }
            finishSubTree(node: result[fold].tree, data: d, fullTrainingSet: d, buildMethod: buildMethod, progress: progress?.foldProgress)
            pruneNode(node: result[fold].tree, data: d)
        
        
            result[fold].confusionMatrix = Array(repeating: Array(repeating: 0, count: data.classes.count), count: data.classes.count)
            if(buildMethod == .OC1) {
                foldSets[fold].fillMissingWith(values: d.averages())
            }
            for p in foldSets[fold].instances {
                let real = data.getClassIndex(value: p.classVal)
                let predict = data.getClassIndex(value: classifyPoint(point: p, dataset: data, classifier: result[fold].tree))
                if let r = real, let p = predict {
                    result[fold].confusionMatrix[p][r] += 1
                }
            }
            progress?.completedFolds += 1
        }
        if(runParallel) {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [i] in
                runFold(fold: i)
                group.leave()
            }
        }
        else {
            runFold(fold: i)
        }
    }
    group.wait()
    
    return result
}
