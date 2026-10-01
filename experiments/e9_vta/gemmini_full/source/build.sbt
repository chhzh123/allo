// Gemmini's whole accelerator, built the way Chipyard builds it but with only
// the projects Gemmini needs: rocket-chip and its three libraries, at the
// commits Gemmini's CHIPYARD.hash pins (Chipyard e0207441, 2025-02-12).
val chiselVersion = "6.5.0"
ThisBuild / scalaVersion := "2.13.12"

lazy val commonSettings = Seq(
  organization := "edu.berkeley.cs",
  scalaVersion := "2.13.12",
  scalacOptions ++= Seq("-deprecation", "-unchecked", "-Ytasty-reader", "-Ymacro-annotations"),
  libraryDependencies += "com.lihaoyi" %% "sourcecode" % "0.3.1",
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value
)

lazy val chiselSettings = Seq(
  libraryDependencies ++= Seq(
    "org.chipsalliance" %% "chisel" % chiselVersion,
    "org.apache.commons" % "commons-lang3" % "3.12.0",
    "org.apache.commons" % "commons-text" % "1.9"
  ),
  addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full)
)

// A project rooted at <dir>/src, so the checkout's own build file is not read.
def freshProject(name: String, dir: File): Project =
  Project(id = name, base = dir / "src").settings(
    Compile / scalaSource := baseDirectory.value / "main" / "scala",
    Compile / resourceDirectory := baseDirectory.value / "main" / "resources"
  )

lazy val midas_target_utils = freshProject("midas_target_utils", file("firesim/sim/midas/targetutils"))
  .settings(commonSettings).settings(chiselSettings)

lazy val cde = freshProject("cde", file("cde/cde"))
  .settings(commonSettings)
  .settings(Compile / scalaSource := baseDirectory.value / "chipsalliance" / "rocketchip")

lazy val hardfloat = freshProject("hardfloat", file("hardfloat/hardfloat"))
  .dependsOn(midas_target_utils)
  .settings(commonSettings).settings(chiselSettings)

lazy val rocketMacros = freshProject("rocketMacros", file("rocket-chip/macros"))
  .settings(commonSettings)

lazy val diplomacy = freshProject("diplomacy", file("diplomacy/diplomacy"))
  .dependsOn(cde)
  .settings(commonSettings).settings(chiselSettings)
  .settings(Compile / scalaSource := baseDirectory.value / "diplomacy")

lazy val rocketchip = freshProject("rocketchip", file("rocket-chip"))
  .dependsOn(hardfloat, rocketMacros, diplomacy, cde)
  .settings(commonSettings).settings(chiselSettings)
  .settings(libraryDependencies ++= Seq(
    "com.lihaoyi" %% "mainargs" % "0.5.0",
    "org.json4s" %% "json4s-jackson" % "4.0.5",
    "org.scala-graph" %% "graph-core" % "1.13.5"
  ))

// Gemmini's three Chipyard SoC configuration files need BOOM and Chipyard.
lazy val gemmini = freshProject("gemmini", file("gemmini"))
  .dependsOn(rocketchip)
  .settings(commonSettings).settings(chiselSettings)
  .settings(Compile / unmanagedSources / excludeFilter :=
    HiddenFileFilter || "CustomCPUConfigs.scala" || "CustomSoCConfigs.scala" || "DSEConfigs.scala")

lazy val harness = (project in file("harness"))
  .dependsOn(gemmini)
  .settings(commonSettings).settings(chiselSettings)
  .settings(run / fork := false)

lazy val root = (project in file(".")).aggregate(harness)
