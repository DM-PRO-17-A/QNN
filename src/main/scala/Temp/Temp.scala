package Temp

import Chisel._
import Modules._

object TempMain {
  def main(args: Array[String]) {
    var cmdArgs = args.slice(1, args.length) 
    println("YAY!")
    chiselMainTest(cmdArgs, () => Module(new Mux2())){ c => new Mux2Tests(c)}
    chiselMainTest(cmdArgs, () => Module(new Max(4,8))){ c => new MaxTests(c)}
  }
}
