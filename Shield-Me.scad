difference(){
import("Blank_IO_Shield-No_Grill.stl");
linear_extrude(height = 5)
    translate([10, 15, -2])
    import("drawing.svg", center=true);
}