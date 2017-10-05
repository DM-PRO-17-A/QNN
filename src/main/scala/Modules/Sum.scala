package Modules

import Chisel._

class Sum(nums_in: Int, dataWidth: Int) extends Module {
	def Adder(a: UInt, b:UInt):UInt = {
		a + b
	}

	val io = new Bundle {
		val nums = Vec.fill(nums_in){UInt(INPUT, dataWidth)}

		val dataOut = UInt(OUTPUT, dataWidth)
	}

	io.dataOut := io.nums.reduceLeft(Adder)
	
}
class SumTests(c: Sum) extends Tester(c) {
	poke(c.io.nums, Array[BigInt](1,2,3,4,5,6,7,8,9))
	peek(c.io.dataOut)
}
