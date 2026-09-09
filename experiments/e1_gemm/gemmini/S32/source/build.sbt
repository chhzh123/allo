name := "gemmini-mesh"
version := "0.1"
scalaVersion := "2.13.10"
addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % "3.6.0" cross CrossVersion.full)
libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.6.0"
// chiseltest 0.6.x is the harness built against chisel 3.6; iotesters 2.5.6 is
// pinned to 3.5.6 and drags in an incompatible json4s.
libraryDependencies += "edu.berkeley.cs" %% "chiseltest" % "0.6.2"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.18"
scalacOptions ++= Seq("-language:reflectiveCalls", "-deprecation", "-feature")
