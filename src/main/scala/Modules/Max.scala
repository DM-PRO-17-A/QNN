package Modules

import Chisel._

class Max(nums_in: Int, dataWidth: Int) extends Module {
  def Compare(a:UInt, b:UInt):UInt = {
    Mux(a > b, a, b)
  }

  val io = new Bundle {
    val nums = Vec.fill(nums_in){UInt(INPUT, dataWidth)}

    val dataOut = UInt(OUTPUT, dataWidth)
  }

  io.dataOut := io.nums.reduceLeft(Compare)
}

class MaxTests(c: Max) extends Tester(c) {
  val input = Array[BigInt](9,14,2,3)
  poke(c.io.nums, input)
  expect(c.io.dataOut, input.reduceLeft(_ max _))
}

