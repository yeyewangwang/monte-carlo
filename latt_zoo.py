from lattpy import simple_square, simple_cubic, simple_hexagonal


def square(shape):
    latt = simple_square()
    latt.build(shape=shape)
    theoretical_tc = 2.7
    return latt

def hexagon(shape):
    latt = simple_hexagonal()
    latt.build(shape=shape)
    return latt

def cubic(shape):
    latt = simple_cubic()
    latt.build(shape=shape)
    return latt
