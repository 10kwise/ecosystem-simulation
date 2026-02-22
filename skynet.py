import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class Grid:
    def __init__(self, width):
        self.width = width
        self.total_tiles = width * width

    def find_neighbours(self, tile):
        neighbours = []
        if tile % self.width != 0:
            neighbours.append(tile - 1)
        if tile % self.width != self.width - 1:
            neighbours.append(tile + 1)
        if tile - self.width >= 0:
            neighbours.append(tile - self.width)
        if tile + self.width < self.total_tiles:
            neighbours.append(tile + self.width)
        return neighbours

def draw_grid(tile):
    ax.clear()
    grid_array = np.zeros((width, width))

    row, col = divmod(tile, width)
    grid_array[row][col] = 1

    for n in grid.find_neighbours(tile):
        nrow, ncol = divmod(n, width)
        grid_array[nrow][ncol] = 0.5

    ax.imshow(grid_array, cmap="Blues")
    ax.set_title(f"tile {tile} and its neighbours")

    # clean up the axes
    ax.set_xticks([])
    ax.set_yticks([])

    # only draw numbers if grid is small enough to read them
    if width <= 20:
        fontsize = max(6, 12 - width // 3)  # scale font down as grid grows
        for i in range(width * width):
            nrow, ncol = divmod(i, width)
            ax.text(ncol, nrow, str(i), ha="center", va="center", fontsize=fontsize)

    canvas.draw()

def on_click():
    try:
        tile = int(entry.get())
        if 0 <= tile < grid.total_tiles:
            draw_grid(tile)
        else:
            label_error.config(text=f"enter a number between 0 and {grid.total_tiles - 1}")
    except ValueError:
        label_error.config(text="please enter a valid number")

def on_width_change():
    global width, grid
    try:
        new_width = int(width_entry.get())
        if new_width < 2:
            label_error.config(text="width must be at least 2")
            return
        width = new_width
        grid = Grid(width)

        # scale figure size with grid, capped so it doesnt get huge
        fig_size = min(10, max(5, width // 5))
        fig.set_size_inches(fig_size, fig_size)

        label_error.config(text="")
        draw_grid(0)
    except ValueError:
        label_error.config(text="please enter a valid width")

width = 3
grid = Grid(width)

window = tk.Tk()
window.title("Grid Visualiser")

fig = Figure(figsize=(5, 5))
ax = fig.add_subplot(111)

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack()

# controls frame
frame = tk.Frame(window)
frame.pack(pady=5)

# width control
tk.Label(frame, text="width:").pack(side=tk.LEFT)
width_entry = tk.Entry(frame, width=5)
width_entry.insert(0, "3")  # default value
width_entry.pack(side=tk.LEFT)
tk.Button(frame, text="set width", command=on_width_change).pack(side=tk.LEFT, padx=5)

# tile control
tk.Label(frame, text="tile:").pack(side=tk.LEFT)
entry = tk.Entry(frame, width=5)
entry.pack(side=tk.LEFT)
tk.Button(frame, text="show neighbours", command=on_click).pack(side=tk.LEFT, padx=5)

# error label
label_error = tk.Label(window, text="", fg="red")
label_error.pack()

draw_grid(0)
window.mainloop()