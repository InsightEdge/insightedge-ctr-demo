name := "insightedge-ctr-demo"

version := "0.3.0"

scalaVersion := "2.10.6"

val insightEdgeVersion = "0.4.0-SNAPSHOT"

val insightEdgeScope = "compile"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "com.gigaspaces.insightedge" % "insightedge-core" % insightEdgeVersion % insightEdgeScope,
  "com.gigaspaces.insightedge" % "gigaspaces-scala" % insightEdgeVersion % insightEdgeScope,
  "com.databricks" %% "spark-csv" % "1.3.0",
  "org.scalatest" %% "scalatest" % "2.0" % "test"
)

//test in assembly := {}

assemblyMergeStrategy in assembly := {
  case PathList("org", "apache", "spark", "unused", "UnusedStubClass.class") => MergeStrategy.first
  case x => (assemblyMergeStrategy in assembly).value(x)
}

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)