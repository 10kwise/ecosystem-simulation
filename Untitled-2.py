width = 3
def find_neighbour2D(coords,width):
    neighbour = [
    (coords[0] - 1,coords[1]),
    (coords[0],coords[1] -1),
    (coords[0] + 1,coords[1]),
    (coords[0],coords[1]+1)
    ]
    return neighbour

def convert_to2D(tile : int):
    x = tile % width
    y = tile // width
    return((x,y))

def convert_to1D(coords : tuple):
    return(coords[1]*width + coords[0])

def find_neighbour1D(tile :int,width : int):
    neighbours = []
    if tile % width != 0:
        neighbours.append(tile-1)
    if tile % width != width - 1:
        neighbours.append(tile + 1)
    if tile - width >= 0 :
        neighbours.append(tile - width)
    if tile + width < width * width:
        neighbours.append(tile + width)
    return neighbours

n = []
tiles = []
for y in range(width):
    for x in range(width):
        coords = (x,y)
        tiles.append(coords)
        neighbours = find_neighbour2D(coords,width)
        tile_number = convert_to1D(coords)
        (reverse_y,reverse_x) = convert_to2D(tile_number)
        reverse_tileno = (reverse_y,reverse_x)
        #print(neighbours)
        #print("coords = " ,coords , "tile_number = " , tile_number , "reversed tile number = " , reverse_tileno,)
        for i in neighbours:
            value = convert_to1D(i)
            if value > 0:
                n.append(value)
            #print(value)

main = convert_to1D(tiles[4])
t = find_neighbour2D(tiles[4],width)
v=[]
for i in t:
    i = convert_to1D(i)
    v.append(i)

print(main)
print(v)
        

        