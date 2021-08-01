import Foundation

protocol Numeric {
    var asDouble: Double { get }
    init(_: Double)
}

extension Int: Numeric {var asDouble: Double { get {return Double(self)}}}
extension Float: Numeric {var asDouble: Double { get {return Double(self)}}}
extension Double: Numeric {var asDouble: Double { get {return Double(self)}}}

extension Array where Element: Numeric {

    var mean : Element? { get { return self.count > 0 ? Element(self.reduce(0, {$0.asDouble + $1.asDouble}) / Double(self.count)) : nil}}

    var sd : Element? { get {
        if(self.isEmpty) {
            return nil
        }
        let mu = self.reduce(0, {$0.asDouble + $1.asDouble}) / Double(self.count)
        let variances = self.map{pow(($0.asDouble - mu), 2)}
        return Element(sqrt(variances.mean!))
    }}
}