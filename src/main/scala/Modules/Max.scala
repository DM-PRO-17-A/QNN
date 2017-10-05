package Modules

import Chisel._

class Max(nums_in: Int, dataWidth: Int) extends Module {
	def Compare(a:UInt, b:UInt):UInt = {
		Mux(a > b, a, b)
	}

	val io = new Bundle {
		val nums = Vec.fill(nums_in){UInt(INPUT, dataWidth)}

		val dataValid = Bool(OUTPUT)
		val dataOut = UInt(OUTPUT, dataWidth)
	}
	
	val currentIndex = Reg(init=(UInt(0,dataWidth)))
	val currentBiggest = Reg(init=(UInt(0,dataWidth)))
	io.dataOut := currentBiggest
	io.dataValid := Bool(false)

	when(currentIndex === UInt(nums_in)){
		io.dataValid := Bool(true)
	} .otherwise {
		currentIndex := currentIndex + UInt(1)
	}

	for(i <- 0 to nums_in-1){
		when(currentIndex === UInt(i)) {
			currentBiggest := Compare(currentBiggest, io.nums(i))
		}
	}

}

class MaxTests(c: Max) extends Tester(c) {
	poke(c.io.nums, Array[BigInt](9,14,2,3))

	for(i <- 0 to 4){
		peek(c.io.dataValid)
		peek(c.io.dataOut)
		step(1)
	}
}

