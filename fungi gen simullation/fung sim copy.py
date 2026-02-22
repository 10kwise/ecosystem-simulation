import random


#genetic info
species_id = []
invasiveness = []
mutation_rate = []
home_court_potency = []
basalmetabolicrate = []
offspring_energy_fraction = []
photosynthetic_ratio = []
spread_threshold = []
species_color = {}

#variable values of organisms

Alive = []
Energy = []
tile_of = []
Age = []

#the world variables 

Width = 200

sun_intensity = 8
max_sunenergy = sun_intensity * 2
base_tile_energy = sun_intensity * 1

initial_organism_count = int((Width * Width)*0.2)
initial_energy = 250
base_energyconsumption = initial_energy * 0.007
spread_energy_threshhold = initial_energy * 1.1

MAX_HISTORY = 4000
CLEANUP_INTERVALS = 1000

max_age = 20000
iterations = 10

World = [None] * Width * Width
Energy_map = [max_sunenergy] * Width * Width
#organism value modifiers and biases
solar_efficiency = 2
triat_costmultiplier = 4
defence_cost_multipliar = 2.5
infectiousness_cost_multipliar = 1.5
invasiveness_bonus_multipliar = 0.01

tau = 0.4
alpha = 3.5

aggresion_reward = 0.2#.4
defense_cost = 0.01

#records
# History tracking for graphs
history_population = []
history_total_energy = []
history_avg_photosynthesis = []
history_avg_defense = []
history_avg_age = []
history_avg_invasiveness = []


next_speciesID = 0
parent_registry = {}


def generate_speciesID(parent_id = None):#parent id is essentially my id if i want to spreadd
    global next_speciesID
    speciesID = next_speciesID
    parent_registry[speciesID] = parent_id
    next_speciesID +=1
    if parent_id is None:
        species_color[speciesID] = (
            random.random(),
            random.random(),
            random.random()
        )
    else:
        parent_color = species_color.get(parent_id)

        # safety fallback if old species lacked colour
        if parent_color is None:
            parent_color = (
                random.random(),
                random.random(),
                random.random()
            )

        species_color[speciesID] = mutate_color(parent_color)
    return speciesID

def calc_BMR(hc_potency,inv_ness,p_ratio):#subject to change by adding age as a consideration
    trait_cost = (hc_potency + inv_ness + p_ratio) * triat_costmultiplier
    

    return base_energyconsumption + (hc_potency * defence_cost_multipliar) + (inv_ness * infectiousness_cost_multipliar) - (p_ratio * solar_efficiency) + trait_cost

def gen_initial_stats():
    id = generate_speciesID()
    s_rate = random.uniform(0.4,1)
    m_rate = random.uniform(0.05,0.2)
    hc_potency = random.uniform(0,1)
    ce_fraction = random.uniform(0.1,0.9)
    p_ratio = random.uniform(0.1,1)
    s_threshold = spread_energy_threshhold * random.uniform(0.9, 1.1)
    staring_point = random.randint(0,(Width * Width)-2)
    tile = None
    species_color[id] = (
    random.random(),
    random.random(),
    random.random()
    )
    while World[staring_point] != None:
        staring_point = random.randint(0,(Width * Width)-2)
    tile = staring_point
    bmr = calc_BMR(hc_potency,s_rate,p_ratio)
    return (initial_energy,id,s_rate,m_rate,hc_potency,ce_fraction,tile,p_ratio,s_threshold,bmr)

def birth_organism(energy : float,s_id,invasive_ness: float,m_rate : float,home_courtpotency : float,ose_fraction :float,tile,p_ratio,s_threshold,o_bmr):
    slot = get_slot()
    Alive[slot] = True
    Energy[slot] = (energy)
    tile_of[slot] = (tile)
    World[tile] = slot
    Age[slot] = (0)

    species_id[slot] = (s_id)
    invasiveness[slot] = (invasive_ness)
    mutation_rate[slot] = (m_rate)
    home_court_potency[slot]=(home_courtpotency)
    basalmetabolicrate[slot]=(o_bmr)
    offspring_energy_fraction[slot] = (ose_fraction)
    photosynthetic_ratio[slot] = (p_ratio)
    spread_threshold[slot] = (s_threshold)

#set up to fix infinite loop growth 
free_slot = []

def kill_organism(i):
    Energy_map[tile_of[i]] += Energy[i]
    Alive[i] = False
    World[tile_of[i]] = None
    tile_of[i] = None
    free_slot.append(i)

def get_slot():

    if free_slot:
        return free_slot.pop()
    
    Alive.append(None)
    Energy.append(None)
    tile_of.append(None)
    Age.append(None)

    species_id.append(None)
    invasiveness.append(None)
    mutation_rate.append(None)
    home_court_potency.append(None)
    basalmetabolicrate.append(None)
    offspring_energy_fraction.append(None)
    photosynthetic_ratio.append(None)
    spread_threshold.append(None)
    return len(Alive) - 1

    
def initialize(organism_count):
    for i in range(organism_count):
        birth_organism(*gen_initial_stats())

def instance(i):#currently working on this is the main loop
    global best_organism_ever , current_best


    Age[i] += 1
    calculate_energychange(i)
    if Energy[i] > 0 and Age[i] <= max_age :
        # do alive things 
        ##        
        if Energy[i] > current_best['energy']:
            current_best['energy'] = Energy[i]
            current_best['energy_index'] = i
        
        if Age[i] > current_best['age']:
            current_best['age'] = Age[i]
            current_best['age_index'] = i
        if Energy[i] > best_organism_ever['energy']:
            best_organism_ever['energy'] = Energy[i]
            best_organism_ever['energy_record_holder'] = {
                'species_id': species_id[i],
                'energy': Energy[i],
                'age': Age[i],
                'photosynthesis': photosynthetic_ratio[i],
                'defense': home_court_potency[i],
                'invasiveness': invasiveness[i],
                'spread_threshold': spread_threshold[i]
            }
        if Age[i] > best_organism_ever['age']:
            best_organism_ever['age'] = Age[i]
            best_organism_ever['age_record_holder'] = {
                'species_id': species_id[i],
                'energy': Energy[i],
                'age': Age[i],
                'photosynthesis': photosynthetic_ratio[i],
                'defense': home_court_potency[i],
                'invasiveness': invasiveness[i],
                'spread_threshold': spread_threshold[i]
            }
        ##
        if Energy[i] > spread_threshold[i]:
            attempt_spread(i)
        
    else:
        kill_organism(i)

def calculate_energychange(i):
    global Energy_map,Energy

    tile = tile_of[i]
    brightness = Energy_map[tile]/base_tile_energy
    gain = max_sunenergy * photosynthetic_ratio[i] * brightness
    Energy_map[tile] -= gain
    Energy[i] += gain - basalmetabolicrate[i]

def update_energy_map():
    global Energy_map

    for i in range(len(Energy_map)):
        Energy_map[i] += sun_intensity

        if Energy_map[i] > base_tile_energy:
            excess = Energy_map[i] - base_tile_energy
            Energy_map[i] -= excess
        

# Records for most successful organisms
best_organism_ever = {
    'energy': 0,
    'age': 0,
    'energy_record_holder': None,  # stores trait snapshot
    'age_record_holder': None
}

current_best = {
    'energy': 0,
    'age': 0,
    'energy_index': None,
    'age_index': None
}

def tick():
    global current_best
    current_best['age'] = 0
    current_best['age_index'] = None
    current_best['energy'] = 0
    current_best['energy_index'] = None
    for i in range(len(Alive)):
        if Alive[i]:
             
            instance(i)
    spread_logic()
    update_energy_map()
    refresh_current_best()

def refresh_current_best():
    current_best['age'] = 0
    current_best['age_index'] = None
    current_best['energy'] = 0
    current_best['energy_index'] = None
    for i in range(len(Alive)):
        if not Alive[i]:
            continue
        if Energy[i] > current_best['energy']:
            current_best['energy'] = Energy[i]
            current_best['energy_index'] = i
        if Age[i] > current_best['age']:
            current_best['age'] = Age[i]
            current_best['age_index'] = i
    

#spread logic 
def reproduce(i, child_energy,new_tile):    
    # Apply mutation or not
    if random.random() <= mutation_rate[i]:
        m_r, hcp, oef, p_r,invasive_ness,s_thresh_delta = mutate(i)
    else:
        m_r = hcp = oef = p_r = invasive_ness = s_thresh_delta = 0  # no mutation
    
    # Clamp traits to valid range
    new_mutation_rate = max(0.01, min(0.5, mutation_rate[i] + m_r))
    new_hcp = max(0, min(1, home_court_potency[i] + hcp))
    new_oef = max(0.1, min(0.9, offspring_energy_fraction[i] + oef))
    new_p_r = max(0.1, min(1.0, photosynthetic_ratio[i] + p_r))
    new_invasiveness = max(0.01,min(1,invasiveness[i] + invasive_ness))
    new_spread_threshold = max(initial_energy * 0.5, min(initial_energy * 2.0, spread_threshold[i] + s_thresh_delta))
    
    bmr = calc_BMR(new_hcp, new_invasiveness, new_p_r)
    birth_organism(child_energy,generate_speciesID(species_id[i]),new_invasiveness,new_mutation_rate,new_hcp,new_oef,new_tile,new_p_r,new_spread_threshold,bmr)
    

pending_births = {}
def attempt_spread(i):    
    neighbours = find_neighbour1D(tile_of[i])
    spread_dir = random.choice(neighbours)
    pending_births.setdefault(spread_dir,set()).add(i)



def calc_winner_e(loser,winner):
    loser_energy = (Energy[loser] * offspring_energy_fraction[loser])
    infectiousness_bonus = (loser_energy *(1 - aggresion_reward)) * invasiveness[winner] * invasiveness_bonus_multipliar
    return infectiousness_bonus + loser_energy * aggresion_reward

def spread_logic():
    for target_tile , organism_indexes in pending_births.items():
        

        strength = {}

        defender = World[target_tile]
        winner = 0
        #attacking empty tile
        if defender == None:
            for index_ in organism_indexes:
                offspring_e = Energy[index_] * offspring_energy_fraction[index_]
                score = (offspring_e) ** tau * ((1 + alpha * invasiveness[index_])) - 1
                strength[index_] = score


            winner = max(strength,key=strength.get)
            reward_for_win = 0 
            for loser in organism_indexes:
                if loser is not winner:                    
                    reward_for_win += calc_winner_e(loser,winner)
            offspring_e_winner = Energy[winner] * offspring_energy_fraction[winner]
            child_e = reward_for_win + offspring_e_winner
            reproduce(winner,child_e,target_tile)
        #attacking occupied tile
        else:

            for index_ in organism_indexes:
                offspring_e = Energy[index_] * offspring_energy_fraction[index_]
                score = (offspring_e/Energy[defender]) ** tau * ((1 + alpha * invasiveness[index_])/(1 + alpha * home_court_potency[defender])) - 1
                strength[index_] = score
            
            winner = max(strength,key=strength.get)

            if max(strength.values()) <= 0:
                Energy[defender] -=len(organism_indexes) * Energy[defender]* defense_cost

            else:
                reward_for_win = 0 
                offspring_e = Energy[winner] * offspring_energy_fraction[winner]
                for loser in organism_indexes:
                    if loser is not winner:                    
                        reward_for_win += calc_winner_e(loser,winner)
                #editting this for smoother dynamic
                # reward_for_win += Energy[defender] * aggresion_reward
                reward_for_win += Energy[defender] * aggresion_reward + (Energy[defender] * invasiveness_bonus_multipliar * (1-aggresion_reward))*invasiveness[winner]
                child_e = reward_for_win + offspring_e
                
                kill_organism(defender)
                reproduce(winner,child_e,target_tile)

        for index_ in organism_indexes:
            offspring_e = Energy[index_] * offspring_energy_fraction[index_]
            Energy[index_] -= offspring_e
    pending_births.clear()




def mutate(i):
        m_r = mutation_rate[i] * mutation_rate[i] * random.choice([1,-1])
        hcp = home_court_potency[i] * mutation_rate[i] * random.choice([1,-1])
        oef = offspring_energy_fraction[i]  * mutation_rate[i] * random.choice([1,-1])
        p_r = photosynthetic_ratio[i]  * mutation_rate[i] * random.choice([1,-1])
        invasive_ness = invasiveness[i] * mutation_rate[i] * random.choice([1,-1])
        s_thresh_delta = spread_threshold[i] * mutation_rate[i] * 0.25 * random.choice([1,-1])
        return m_r, hcp,oef,p_r,invasive_ness,s_thresh_delta
        
def mutate_color(parent):
    r,g,b = parent
    drift = 0.05
    return (
        min(1,max(0,r + random.uniform(-drift,drift))),
        min(1,max(0,g + random.uniform(-drift,drift))),
        min(1,max(0,b + random.uniform(-drift,drift)))
    )

            
#visualisations
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def setup_visualization():
    global main_window, grid_window, canvas, grid_canvas, im, e_im
    global grid_ax, e_map_ax, pop_ax, energy_ax, trait_ax
    global pop_line, energy_line, photo_line, defense_line,invasive_line,stats_text
    # main window for graphs
    main_window = tk.Tk()
    main_window.title("Statistics")
    # main_window.state('zoomed')
    
    # separate window just for grid
    grid_window = tk.Toplevel(main_window)
    grid_window.title("World Grid")
    # grid_window.state('zoomed')
    
    # grid figure in its own window
    grid_fig = Figure(figsize=(12, 6))
    grid_ax = grid_fig.add_subplot(121)
    e_map_ax = grid_fig.add_subplot(122)
    grid_ax.axis('off')
    e_map_ax.axis('off')
    grid_ax.set_title("Organisms")
    e_map_ax.set_title("Energy Map")
    world_array = np.zeros((Width, Width, 3))
    e_map_arr = np.zeros((Width, Width))
    im = grid_ax.imshow(world_array, interpolation='nearest')
    e_im = e_map_ax.imshow(e_map_arr, vmin=0, vmax=1, cmap='magma', interpolation='nearest')
    grid_canvas = FigureCanvasTkAgg(grid_fig, master=grid_window)
    grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    grid_fig.tight_layout()
    grid_window.lift()
    grid_window.attributes('-topmost', True)
    fig = Figure(figsize=(15, 10))
    
    # Grid visualization
    # grid_ax = fig.add_subplot(2, 3, 1)
    # grid_ax.set_title("World Grid")
    # grid_ax.axis('off')
    
    # world_array = np.zeros((width, width, 3))
    # im = grid_ax.imshow(world_array)
    
    # Population over time (top right)
    pop_ax = fig.add_subplot(2, 3, 6)
    pop_ax.set_title("Population Over Time")
    pop_ax.set_ylim(0, Width * Width)
    pop_ax.set_xlabel("Tick")
    pop_ax.set_ylabel("Population")
    pop_line, = pop_ax.plot([], [], 'b-')
    
    # Total energy over time (middle right)
    energy_ax = fig.add_subplot(2, 3, 4)
    energy_ax.set_title("Total Energy Over Time")
    if history_total_energy:
        energy_ax.set_ylim(0, max(history_total_energy) * 1.1)  # 10% headroom
    energy_ax.set_xlabel("Tick")
    energy_ax.set_ylabel("Total Energy")
    energy_line, = energy_ax.plot([], [], 'r-')
    
    # Average traits over time (bottom)
    trait_ax = fig.add_subplot(2, 3, 5)  # spans full bottom
    trait_ax.set_title("Average Traits Over Time")
    trait_ax.set_ylim(0, 1)
    trait_ax.set_xlabel("Tick")
    trait_ax.set_ylabel("Trait Value")

    # Add after creating trait_ax
    stats_ax = fig.add_subplot(2, 3, 3)
    stats_ax.axis('off')  # Hide axes for text display
    stats_text = stats_ax.text(0.05, 0.95, '', transform=stats_ax.transAxes,
                            fontsize=9, verticalalignment='top', family='monospace')
    
    photo_line, = trait_ax.plot([], [], 'g-', label='Photosynthesis')
    defense_line, = trait_ax.plot([], [], 'b-', label='Defense')
    invasive_line, = trait_ax.plot([], [], 'r-', label='Invasiveness')
    trait_ax.legend()
    
    pop_ax.grid(True, alpha=0.3)  # subtle grid
    energy_ax.grid(True, alpha=0.3)
    trait_ax.grid(True, alpha=0.3)
    trait_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.tight_layout()  # automatically adjust spacing
    
    canvas = FigureCanvasTkAgg(fig, master=main_window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # fill window

def downsample(data , max_points = 2000):#reads the data in steps to reduce number of datapoints plotted
    if len(data) <= max_points:
        return range(len(data)) , data
    
    step = len(data) // max_points
    return range(0,len(data),step) , data[::step]

def cleanup_extinct_lineage():
    living_species = set(species_id[i] for i in range(len(Alive)) if Alive[i])
    extinct = set(species_color.keys()) - living_species
    for s_id in extinct:
        del species_color[s_id]
        if s_id in parent_registry:
            del parent_registry[s_id]




def update_visualization():
    # run one or more ticks
    for i in range(iterations):
        tick()
        collect_statistics()

    # rebuild world array
    world_array = np.zeros((Width, Width, 3))
    energy_map_arr = np.zeros((Width, Width))
    for i in range(len(Energy_map)):
        x, y = convert_to2D(i)
        energy_map_arr[y][x] = Energy_map[i] / base_tile_energy

    for i in range(len(Alive)):
        if not Alive[i]:
            continue

        x, y = convert_to2D(tile_of[i])
        R,G,B = species_color[species_id[i]]
        Br = min(1,Age[i]/max_age)+ 0.5
        RGB = (R*Br,G*Br,B*Br)
        world_array[y][x] = RGB
    
    # update the existing image
    world_array = np.clip(world_array, 0, 1)
    energy_map_arr = np.clip(energy_map_arr, 0, 1)
    im.set_data(world_array)
    e_im.set_data(energy_map_arr)

    # Update line plots
    x , y = downsample(history_population)
    pop_line.set_data(x,y)

    x , y = downsample(history_total_energy)
    energy_line.set_data(x,y)

    x , y = downsample(history_avg_photosynthesis)
    photo_line.set_data(x,y)

    x , y = downsample(history_avg_defense)
    defense_line.set_data(x,y)

    x , y = downsample(history_avg_invasiveness)
    invasive_line.set_data(x,y)

    stats_text.set_text(generate_stats_text())
    
    # Rescale axes to fit new data
    pop_ax.relim()
    pop_ax.autoscale_view()
    energy_ax.relim()
    energy_ax.autoscale_view()
    trait_ax.relim()
    trait_ax.autoscale_view()

    im.set_data(world_array)
    grid_canvas.draw()  # explicitly draw grid
    grid_canvas.flush_events()  # force immediate update
    
    canvas.draw()  # draw stats
    canvas.flush_events()
    # schedule next update (50ms = 20fps)
    main_window.after(50, update_visualization)



def collect_statistics():
    living = [i for i in range(len(Alive)) if Alive[i]]
    history_population.append(len(living))
    history_total_energy.append(sum(Energy[i] for i in living))
    if living:  # avoid division by zero
        history_avg_photosynthesis.append(sum(photosynthetic_ratio[i] for i in living) / len(living))
        history_avg_defense.append(sum(home_court_potency[i] for i in living) / len(living))
        history_avg_age.append(sum(Age[i] for i in living) / len(living))
        history_avg_invasiveness.append(sum(invasiveness[i] for i in living) / len(living))
    else:
        history_avg_photosynthesis.append(0)
        history_avg_defense.append(0)
        history_avg_age.append(0)
        history_avg_invasiveness.append(0) 

    if len(history_population) % CLEANUP_INTERVALS == 0:
        trim_history()

#optimization to reduce amount of ticks being tracked
def trim_history():
    global history_population, history_total_energy , history_avg_age,history_avg_photosynthesis,history_avg_defense,history_avg_invasiveness

    if len(history_population) > MAX_HISTORY:
        trim = len(history_population) - MAX_HISTORY

        history_population = history_population[trim:]
        history_total_energy = history_total_energy[trim:]
        history_avg_photosynthesis = history_avg_photosynthesis[trim:]
        history_avg_defense = history_avg_defense[trim:]
        history_avg_age = history_avg_age[trim:]
        history_avg_invasiveness = history_avg_invasiveness[trim:]


def generate_stats_text():
    text = "=== CURRENT BEST ===\n"
    
    if current_best['energy_index'] is not None:
        i = current_best['energy_index']
        text += f"Highest Energy: {Energy[i]:.0f}\n"
        text += f"  Age: {Age[i]}, Species: {species_id[i]}\n"
        text += f"  P:{photosynthetic_ratio[i]:.2f} D:{home_court_potency[i]:.2f} I:{invasiveness[i]:.2f} S:{spread_threshold[i]:.0f}\n\n"
    
    if current_best['age_index'] is not None:
        i = current_best['age_index']
        text += f"Oldest: {Age[i]} ticks\n"
        text += f"  Energy: {Energy[i]:.0f}, Species: {species_id[i]}\n"
        text += f"  P:{photosynthetic_ratio[i]:.2f} D:{home_court_potency[i]:.2f} I:{invasiveness[i]:.2f} S:{spread_threshold[i]:.0f}\n\n"
    
    text += "=== ALL-TIME RECORDS ===\n"
    if best_organism_ever['energy_record_holder']:
        rec = best_organism_ever['energy_record_holder']
        text += f"Energy Record: {rec['energy']:.0f}\n"
        text += f"  Age: {rec['age']}, Species: {rec['species_id']}\n"
        text += f"  P:{rec['photosynthesis']:.2f} D:{rec['defense']:.2f} I:{rec['invasiveness']:.2f} S:{rec['spread_threshold']:.0f}\n\n"
    
    if best_organism_ever['age_record_holder']:
        rec = best_organism_ever['age_record_holder']
        text += f"Age Record: {rec['age']} ticks\n"
        text += f"  Energy: {rec['energy']:.0f}, Species: {rec['species_id']}\n"
        text += f"  P:{rec['photosynthesis']:.2f} D:{rec['defense']:.2f} I:{rec['invasiveness']:.2f} S:{rec['spread_threshold']:.0f}"
    
    return text



def start():
    initialize(initial_organism_count)
    setup_visualization()
    update_visualization()
    


#helper functions
def find_neighbour1D(tile :int):
    neighbours = []
    if tile % Width != 0:
        neighbours.append(tile-1)
    if tile % Width != Width - 1:
        neighbours.append(tile + 1)
    if tile - Width >= 0 :
        neighbours.append(tile - Width)
    if tile + Width < Width * Width:
        neighbours.append(tile + Width)
    return neighbours


def convert_to2D(tile : int):
    y, x = divmod(tile, Width)
    return (x, y)

def convert_to1D(coords : tuple):
    return(coords[1]*Width + coords[0])




start()# start the update loop
main_window.mainloop()



