$outset_x  = 20;
$outset_y  = 30;
$outset_y2 = 35;
$outset_mag= 37;

$outer_r = 55;

union() {

    difference() {
        union() {
            cylinder(h=4,r=55,center=false);
            cylinder(h=6,r=$outer_r - 5,center=false);
        }
        union() {
            translate([-$outset_x,$outset_y,0]) {
                cylinder(h=25,r=5.5,center=true); // Loch
            }
            translate([$outset_x,$outset_y,0]) {
                cylinder(h=25,r=5.5,center=true); // Loch
            }
            translate([0,-$outset_y2,0]) {
                cylinder(h=25,r=5.5,center=true); // Loch
            }

            translate([-$outset_mag,0,3]) {
                cylinder(h=10,r=16/2,center=false); // Magnet
                cylinder(h=25,r=2.1,center=true); // M4 Loch
            }
            translate([$outset_mag,0,3]) {
                cylinder(h=10,r=16/2,center=false); // Magnet
                cylinder(h=25,r=2.1,center=true); // M4 Loch
            }
            cylinder(h=25,r=25,center=true); // Loch, mitte
            
            for(rot =[0:5])
            {
                rotate([0,0,rot*60])
                {
                    translate([0,$outer_r-7,0]) {
                        cube([8,4,25],center=true);
                    }
                }
            }
        }
    }

    translate([-$outset_x,-$outset_y,0]) {
        translate([0,0,11]) {
            cylinder(h=4,r1=5,r2=3.5,center=false);
        }
        cylinder(h=11,r=5,center=false);
    }
    translate([$outset_x,-$outset_y,0]) {
        translate([0,0,11]) {
            cylinder(h=4,r1=5,r2=3.5,center=false);
        }
        cylinder(h=11,r=5,center=false);
    }
    translate([0,$outset_y2,0]) {
        translate([0,0,11]) {
            cylinder(h=4,r1=5,r2=3.5,center=false);
        }
        cylinder(h=11,r=5,center=false);
    }

}
