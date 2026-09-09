name := "gemmini-mesh"
version := "0.1"
scalaVersion := "2.13.10"
addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % "3.6.0" cross CrossVersion.full)
libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.6.0"
libraryDependencies += "edu.berkeley.cs" %% "chisel-iotesters" % "2.5.6" % "test"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.18" % "test"
scalacOptions ++= Seq("-language:reflectiveCalls", "-deprecation", "-feature")
