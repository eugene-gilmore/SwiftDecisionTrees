// swift-tools-version:5.0
import PackageDescription

let package = Package(
	name: "SwiftDecisionTrees",
	products: [
		.library(name: "SwiftDecisionTrees", targets: ["SwiftDecisionTrees"])
	],
	dependencies: [
		.package(url: "https://github.com/IBM-Swift/BlueSignals.git", .branch("master")),
		.package(url: "https://github.com/eugene-gilmore/JSONCodable.git", .branch("master")),
	],
	targets : [
		.target(name: "OC1", dependencies: []),
		.target(name: "SwiftDecisionTrees", dependencies: ["JSONCodable", .target(name: "OC1")]),
		.target(name: "DecisionTreeDemo", dependencies: [.target(name: "SwiftDecisionTrees"), "Signals"])
	]
)

