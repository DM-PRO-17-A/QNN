package Modules

import Chisel._

class ScaleShift(array_size: Int, data_width) extends Module {
    
    val io = new Bundle {
        val nums_in = Vec.fill(array_size){UInt(INPUT, data_width)}
        
        val data_out = Vec.fill(array_size){UInt(OUTPUT, data_width)}
    }

    val scale = Module(new Scale(array_size, data_width)).io
    val shift = Module(new Shift(array_size, data_width)).io

    scale.nums_in := io.nums_in
    shift.nums_in := scale.nums_out

    // Weights?
    
    io.data_out := shift.data_out
    
}
