stage 'Build'
env.PATH = "${tool 'sbt-0.13.11'}/bin:$env.PATH"
load 'tools/build.groovy'