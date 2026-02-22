import random
import colorsys
import os
import pickle
import json
import sys
import time
import tracemalloc
from datetime import datetime

#genetic info
species_id = []
invasiveness = []
mutation_rate = []
home_court_potency = []
basalmetabolicrate = []
offspring_energy_fraction = []
photosynthetic_ratio = []
spread_threshold = []
parasitism_rate = []  # 0-1 trait
species_color = {}

#variable values of organisms

Alive = []
Energy = []
tile_of = []
Age = []

#the world variables 

Width = 50

sun_intensity = 8
max_sunenergy = sun_intensity * 2
base_tile_energy = sun_intensity * 1

initial_organism_count = int((Width * Width)*0.5)
initial_energy = 150
base_energyconsumption = initial_energy * 0.007
spread_energy_threshhold = initial_energy * 1.1

MAX_HISTORY = 8000
CLEANUP_INTERVALS = 2000

max_age = 200000
run_ticks = 5000  # adjust this to control simulation length before replay
replay_interval_ms = 50

World = [None] * Width * Width
Energy_map = [max_sunenergy] * Width * Width
neighbor_cache = []
spatial_occupancy = {}

#organism value modifiers and biases
solar_efficiency = 2.6
triat_costmultiplier = 3.2
defence_cost_multipliar = 6
invasiveness_cost_multipliar = 2
invasiveness_bonus_multipliar = 0.006
parasitism_cost_multiplier = 4
parasitism_flat_drain = 3.0  # flat energy drain at parasitism=1.0 before mitigation

tau = 0.5
alpha = 2.1

aggresion_reward = 0.12#.4
defense_cost = 0.004

#records
# History tracking for graphs
history_population = []
history_total_energy = []
history_avg_photosynthesis = []
history_avg_defense = []
history_avg_age = []
history_avg_invasiveness = []
history_avg_parasitism = []
history_avg_gain = []
history_avg_bmr = []
history_combat_income = []
history_species_richness = []
history_shannon_diversity = []
history_evenness = []
history_age_structure = []
history_energy_gini = []
history_territory_stability = []
history_trait_investment = []
history_trait_corr = []
history_attacker_win_rate = []
history_para_photo_ratio = []

# in-memory run archive + replay
run_archive = []
current_run_frames = []
current_capture_step = 0
MAX_REPLAY_FRAMES = 2000
FRAME_CAPTURE_STRIDE = 5
MAX_ARCHIVED_RUNS = 8
replay_mode = False
replay_run_index = -1
replay_frame_index = 0
replay_playing = True

runs_dir = os.path.join(os.path.dirname(__file__), "saved_runs")
analysis_dir = os.path.join(os.path.dirname(__file__), "analysis_exports")
age_bins = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, max_age + 1)]

# profiling/export controls
ENABLE_AI_PROFILING = True
EXPORT_CODE_SNAPSHOT = True

# per-tick phase timings (ms)
tick_phase_instance_ms = 0.0
tick_phase_spread_ms = 0.0
tick_phase_energymap_ms = 0.0
tick_phase_refresh_ms = 0.0
tick_phase_total_ms = 0.0

# Environment presets (variables only) to explore coexistence/symbiosis
BASE_ENVIRONMENT = {
    "sun_intensity": sun_intensity,
    "solar_efficiency": solar_efficiency,
    "triat_costmultiplier": triat_costmultiplier,
    "defence_cost_multipliar": defence_cost_multipliar,
    "invasiveness_cost_multipliar": invasiveness_cost_multipliar,
    "invasiveness_bonus_multipliar": invasiveness_bonus_multipliar,
    "parasitism_cost_multiplier": parasitism_cost_multiplier,
    "parasitism_flat_drain": parasitism_flat_drain,
    "tau": tau,
    "alpha": alpha,
    "aggresion_reward": aggresion_reward,
    "defense_cost": defense_cost,
}

ENVIRONMENT_PRESETS = {
    "baseline": {},
    # Lower combat snowball + moderate parasite drain supports longer co-existence.
    "symbiosis_soft_competition": {
        "sun_intensity": 9.0,
        "solar_efficiency": 2.9,
        "triat_costmultiplier": 2.8,
        "invasiveness_bonus_multipliar": 0.0025,
        "parasitism_flat_drain": 0.8,
        "tau": 0.58,
        "alpha": 1.4,
        "aggresion_reward": 0.05,
        "defense_cost": 0.010,
    },
    # Patchy coexistence: some conflict, but less wipeout pressure.
    "symbiosis_patchy": {
        "sun_intensity": 8.0,
        "solar_efficiency": 2.8,
        "triat_costmultiplier": 3.0,
        "invasiveness_bonus_multipliar": 0.0030,
        "parasitism_flat_drain": 1.1,
        "tau": 0.55,
        "alpha": 1.6,
        "aggresion_reward": 0.07,
        "defense_cost": 0.008,
    },
    # Tense but stable: parasites remain meaningful without immediate collapse.
    "symbiosis_tense": {
        "sun_intensity": 8.5,
        "solar_efficiency": 2.7,
        "triat_costmultiplier": 3.1,
        "invasiveness_bonus_multipliar": 0.0035,
        "parasitism_flat_drain": 1.4,
        "tau": 0.53,
        "alpha": 1.7,
        "aggresion_reward": 0.08,
        "defense_cost": 0.007,
    },
}

ENVIRONMENT_PRESET = "symbiosis_tense"


next_speciesID = 0
parent_registry = {}

def refresh_environment_derived_values():
    global max_sunenergy, base_tile_energy
    max_sunenergy = sun_intensity * 2
    base_tile_energy = sun_intensity * 1


def apply_environment_preset(name):
    global sun_intensity, solar_efficiency, triat_costmultiplier, defence_cost_multipliar
    global invasiveness_cost_multipliar, invasiveness_bonus_multipliar
    global parasitism_cost_multiplier, parasitism_flat_drain, tau, alpha
    global aggresion_reward, defense_cost

    if name not in ENVIRONMENT_PRESETS:
        raise ValueError(f"Unknown environment preset: {name}")

    config = dict(BASE_ENVIRONMENT)
    config.update(ENVIRONMENT_PRESETS[name])
    for key, value in config.items():
        globals()[key] = value
    refresh_environment_derived_values()


def reset_energy_map():
    for i in range(len(Energy_map)):
        Energy_map[i] = max_sunenergy


def generate_speciesID(parent_id = None):#parent id is essentially my id if i want to spreadd
    global next_speciesID
    speciesID = next_speciesID
    parent_registry[speciesID] = parent_id
    next_speciesID +=1

    if parent_id is None:
        # Founding organism - assign distinct color from palette
        hue = random.random()  # 0-1 for HSV
        species_color[speciesID] = {
            'hue': hue,
            'base_hue': hue,  # Remember original lineage color
            'generation': 0
        }
    else:
        parent_color = species_color.get(parent_id)
        if parent_color:
            # Inherit BASE hue from parent (tracks lineage)
            # But allow small drift in actual hue
            drift = random.uniform(-0.05, 0.05)  # Small drift
            new_hue = (parent_color['hue'] + drift) % 1.0
            
            species_color[speciesID] = {
                'hue': new_hue,
                'base_hue': parent_color['base_hue'],  # Keep lineage marker
                'generation': parent_color['generation'] + 1
            }
    
    return speciesID

def calc_BMR(hc_potency,inv_ness,p_ratio,parasitism_r):#subject to change by adding age as a consideration
    trait_cost = (hc_potency + inv_ness + p_ratio + parasitism_r) * triat_costmultiplier
    

    return base_energyconsumption + (hc_potency * defence_cost_multipliar) + (inv_ness * invasiveness_cost_multipliar) - (p_ratio * solar_efficiency) + trait_cost + (parasitism_r * parasitism_cost_multiplier)

def gen_initial_stats():
    id = generate_speciesID()
    s_rate = random.uniform(0.4,1)
    m_rate = random.uniform(0.05,0.2)
    hc_potency = random.uniform(0,1)
    ce_fraction = random.uniform(0.1,0.9)
    p_ratio = random.uniform(0.1,1)#for photosynthesis
    s_threshold = spread_energy_threshhold * random.uniform(0.9, 1.1)
    staring_point = random.randint(0,(Width * Width)-2)
    tile = None
    p_rate = random.uniform(0, 0.5)#for parasitsm
    while World[staring_point] != None:
        staring_point = random.randint(0,(Width * Width)-2)
    tile = staring_point
    bmr = calc_BMR(hc_potency,s_rate,p_ratio,p_rate)
    return (initial_energy,id,s_rate,m_rate,hc_potency,ce_fraction,tile,p_ratio,s_threshold,bmr,p_rate)

def birth_organism(energy : float,s_id,invasive_ness: float,m_rate : float,home_courtpotency : float,ose_fraction :float,tile,p_ratio,s_threshold,o_bmr,p_rate_param):
    slot = get_slot()
    Alive[slot] = True
    Energy[slot] = (energy)
    tile_of[slot] = (tile)
    World[tile] = slot
    spatial_occupancy[tile] = slot
    Age[slot] = (0)
    parasitism_rate[slot] = p_rate_param
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
    victim_tile = tile_of[i]
    Energy_map[victim_tile] += Energy[i]
    Alive[i] = False
    World[victim_tile] = None
    spatial_occupancy.pop(victim_tile, None)
    tile_of[i] = None
    free_slot.append(i)

def get_slot():

    if free_slot:
        return free_slot.pop()
    
    Alive.append(None)
    Energy.append(None)
    tile_of.append(None)
    Age.append(None)
    parasitism_rate.append(None)
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
    handle_overflow(i)
    calculate_parasitism(i)
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
                'spread_threshold': spread_threshold[i],
                'parasitism': parasitism_rate[i]
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
                'spread_threshold': spread_threshold[i],
                'parasitism': parasitism_rate[i]
            }
        ##
        if Energy[i] > spread_threshold[i]:
            attempt_spread(i)
        
    else:
        kill_organism(i)
def calculate_size_based_max_energy(i):
    # Larger defense = larger organism = more storage
    # Could also factor in age: older = larger
    size_factor = 1.0 + (home_court_potency[i] * 2.0)  # 1.0 to 3.0 range
    return initial_energy * size_factor

def handle_overflow(i):
    max_e = calculate_size_based_max_energy(i)
    
    if Energy[i] > max_e:
        overflow = Energy[i] - max_e
        Energy[i] = max_e
        
        # Leak to neighboring tiles
        neighbors = find_neighbour1D(tile_of[i])
        per_neighbor = overflow / len(neighbors)
        for n_tile in neighbors:
            Energy_map[n_tile] += per_neighbor
def calculate_energychange(i):
    global Energy_map, Energy, tick_total_gain, tick_total_bmr, tick_metabolism_count

    tile = tile_of[i]
    brightness = Energy_map[tile]/base_tile_energy
    gain = max_sunenergy * photosynthetic_ratio[i] * brightness
    Energy_map[tile] -= gain
    Energy[i] += gain - basalmetabolicrate[i]
    tick_total_gain += gain
    tick_total_bmr += basalmetabolicrate[i]
    tick_metabolism_count += 1

def calculate_parasitism(parasite_idx):
    global tick_parasitism_income
    tile = tile_of[parasite_idx]
    neighbors = find_neighbour1D(tile)
    total_drained = 0
    
    for n_tile in neighbors:
        victim_idx = spatial_occupancy.get(n_tile)
        if victim_idx is None:
            continue  # Empty tile
        
        # Base percent + flat drain components
        base_drain_percent = parasitism_rate[parasite_idx] * 0.2
        base_flat_drain = parasitism_rate[parasite_idx] * parasitism_flat_drain
        
        # Victim's defense reduces drain
        defense_reduction = home_court_potency[victim_idx] * 0.35
        
        # Victim's own parasitism provides resistance
        parasite_resistance = parasitism_rate[victim_idx] * 0.3
        
        # Net percent drain (never negative)
        net_drain_percent = max(0, base_drain_percent - defense_reduction - parasite_resistance)

        # Flat drain uses same mitigation path as percent drain
        flat_mitigation = max(0, 1 - defense_reduction - parasite_resistance)
        net_flat_drain = base_flat_drain * flat_mitigation

        # Drain from victim (cannot exceed victim energy)
        energy_drained = (Energy[victim_idx] * net_drain_percent) + net_flat_drain
        energy_drained = min(energy_drained, max(0, Energy[victim_idx]))
        if energy_drained <= 0:
            continue
        Energy[victim_idx] -= energy_drained
        Energy[parasite_idx] += energy_drained
        total_drained += energy_drained
        tick_parasitism_income += energy_drained
    
    return total_drained
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

# per-tick debug metrics
tick_total_gain = 0.0
tick_total_bmr = 0.0
tick_metabolism_count = 0
tick_combat_income = 0.0
tick_parasitism_income = 0.0
tick_total_combats = 0
tick_attacker_wins = 0
previous_tile_species = [None] * (Width * Width)

def tick():
    global current_best, tick_total_gain, tick_total_bmr, tick_metabolism_count, tick_combat_income
    global tick_parasitism_income, tick_total_combats, tick_attacker_wins
    global tick_phase_instance_ms, tick_phase_spread_ms, tick_phase_energymap_ms
    global tick_phase_refresh_ms, tick_phase_total_ms

    t0 = time.perf_counter()
    current_best['age'] = 0
    current_best['age_index'] = None
    current_best['energy'] = 0
    current_best['energy_index'] = None
    tick_total_gain = 0.0
    tick_total_bmr = 0.0
    tick_metabolism_count = 0
    tick_combat_income = 0.0
    tick_parasitism_income = 0.0
    tick_total_combats = 0
    tick_attacker_wins = 0
    for i in range(len(Alive)):
        if Alive[i]:
             
            instance(i)
    t1 = time.perf_counter()
    spread_logic()
    t2 = time.perf_counter()
    update_energy_map()
    t3 = time.perf_counter()
    refresh_current_best()
    t4 = time.perf_counter()

    tick_phase_instance_ms = (t1 - t0) * 1000
    tick_phase_spread_ms = (t2 - t1) * 1000
    tick_phase_energymap_ms = (t3 - t2) * 1000
    tick_phase_refresh_ms = (t4 - t3) * 1000
    tick_phase_total_ms = (t4 - t0) * 1000

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
        mutations = mutate(i)
        m_r, hcp, oef, p_r, invasive_ness, s_thresh_delta, p_rate_delta = mutations
        # Normalize threshold-delta so one large-scale trait does not dominate speciation.
        normalized_s_thresh_delta = abs(s_thresh_delta) / max(initial_energy * 5.0, 1.0)
        mutation_magnitude = (
            abs(m_r)
            + abs(hcp)
            + abs(oef)
            + abs(p_r)
            + abs(invasive_ness)
            + normalized_s_thresh_delta
            + abs(p_rate_delta)
        )

        if mutation_magnitude > 0.3:  # Threshold for new species
            child_species = generate_speciesID(species_id[i])
        else:
            child_species = species_id[i]  # Small variation keeps parent species
    else:
        child_species = species_id[i]  # No mutation = same species
        m_r = hcp = oef = p_r = invasive_ness = s_thresh_delta=p_rate_delta = 0  # no mutation
    
    # Clamp traits to valid range
    new_mutation_rate = max(0.01, min(0.5, mutation_rate[i] + m_r))
    new_hcp = max(0, min(1, home_court_potency[i] + hcp))
    new_oef = max(0.1, min(0.9, offspring_energy_fraction[i] + oef))
    new_p_r = max(0.1, min(1.0, photosynthetic_ratio[i] + p_r))
    new_invasiveness = max(0.01,min(1,invasiveness[i] + invasive_ness))
    new_spread_threshold = max(initial_energy * 0.2, min(initial_energy * 5.0, spread_threshold[i] + s_thresh_delta))
    new_parasitism = max(0, min(1, parasitism_rate[i] + p_rate_delta))
    
    bmr = calc_BMR(new_hcp, new_invasiveness, new_p_r,new_parasitism)
    birth_organism(child_energy,child_species,new_invasiveness,new_mutation_rate,new_hcp,new_oef,new_tile,new_p_r,new_spread_threshold,bmr,new_parasitism)
    

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
    global tick_combat_income, tick_total_combats, tick_attacker_wins
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
            tick_combat_income += reward_for_win
            child_e = reward_for_win + offspring_e_winner
            reproduce(winner,child_e,target_tile)
        #attacking occupied tile
        else:
            tick_total_combats += 1
            # Defender can reach 0 energy from earlier per-tick effects; avoid divide-by-zero.
            defender_energy = max(Energy[defender], 1e-9)

            for index_ in organism_indexes:
                offspring_e = Energy[index_] * offspring_energy_fraction[index_]
                score = (offspring_e / defender_energy) ** tau * ((1 + alpha * invasiveness[index_])/(1 + alpha * home_court_potency[defender])) - 1
                strength[index_] = score
            
            winner = max(strength,key=strength.get)

            if max(strength.values()) <= 0:
                Energy[defender] -=len(organism_indexes) * Energy[defender]* defense_cost

            else:
                tick_attacker_wins += 1
                reward_for_win = 0 
                offspring_e = Energy[winner] * offspring_energy_fraction[winner]
                for loser in organism_indexes:
                    if loser is not winner:                    
                        reward_for_win += calc_winner_e(loser,winner)
                #editting this for smoother dynamic
                # reward_for_win += Energy[defender] * aggresion_reward
                reward_for_win += Energy[defender] * aggresion_reward + (Energy[defender] * invasiveness_bonus_multipliar * (1-aggresion_reward))*invasiveness[winner]
                tick_combat_income += reward_for_win
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
    p_rate_delta = parasitism_rate[i] * mutation_rate[i] * random.choice([1,-1])
    return m_r, hcp, oef, p_r, invasive_ness, s_thresh_delta, p_rate_delta
        
def mutate_color(parent,p_id):
    r,g,b = parent
    drift = 0.05
    return (
        min(1,max(0,r + random.uniform(-drift,drift))),
        min(1,max(0,g + random.uniform(-drift,drift))),
        min(1,max(0,b + random.uniform(-drift,drift)))
    )


def classify_guild_from_traits(photo, invasive, parasite):
    if parasite >= photo and parasite >= invasive and parasite >= 0.25:
        return "parasite"
    if photo >= invasive and photo >= parasite and photo >= 0.25:
        return "plant"
    if invasive >= photo and invasive >= parasite and invasive >= 0.25:
        return "invasive"
    return "mixed"


def classify_guild(i):
    return classify_guild_from_traits(
        photosynthetic_ratio[i],
        invasiveness[i],
        parasitism_rate[i]
    )


def get_current_tile_species():
    tile_species = [None] * (Width * Width)
    for tile_idx, org_idx in enumerate(World):
        if org_idx is not None and Alive[org_idx]:
            tile_species[tile_idx] = species_id[org_idx]
    return tile_species


def calculate_energy_gini(energies):
    if not energies:
        return 0.0
    sorted_energies = sorted(energies)
    n = len(sorted_energies)
    total_energy = sum(sorted_energies)
    if total_energy <= 0:
        return 0.0

    weighted_sum = 0.0
    for i, e in enumerate(sorted_energies, start=1):
        weighted_sum += (2 * i - n - 1) * e
    gini = weighted_sum / (n * total_energy)
    return max(0.0, min(1.0, gini))


def calculate_trait_correlation(living):
    if len(living) < 2:
        return np.eye(4, dtype=float)

    arr = np.array([
        [photosynthetic_ratio[i], home_court_potency[i], invasiveness[i], parasitism_rate[i]]
        for i in living
    ], dtype=float)
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def calculate_age_structure(living):
    counts = [0] * len(age_bins)
    for i in living:
        a = Age[i]
        for b_idx, (lo, hi) in enumerate(age_bins):
            if lo <= a < hi:
                counts[b_idx] += 1
                break

    total = len(living)
    if total == 0:
        return [0.0] * len(age_bins)
    return [c / total for c in counts]


def compute_observations(living):
    global previous_tile_species

    energies = [Energy[i] for i in living]
    age_structure = calculate_age_structure(living)
    gini = calculate_energy_gini(energies)

    species_counts = {}
    for i in living:
        sid = species_id[i]
        species_counts[sid] = species_counts.get(sid, 0) + 1
    richness = len(species_counts)
    shannon = 0.0
    total = len(living)
    if total > 0:
        for c in species_counts.values():
            p = c / total
            if p > 0:
                shannon -= p * np.log(p)
    species_evenness = (shannon / np.log(richness)) if richness > 1 else 0.0

    current_tile_species = get_current_tile_species()
    unchanged = 0
    for prev, curr in zip(previous_tile_species, current_tile_species):
        if prev == curr:
            unchanged += 1
    territory_stability = unchanged / (Width * Width)
    previous_tile_species = current_tile_species

    if living:
        trait_investment = sum(
            photosynthetic_ratio[i] + home_court_potency[i] + invasiveness[i] + parasitism_rate[i]
            for i in living
        ) / len(living)
    else:
        trait_investment = 0.0

    corr = calculate_trait_correlation(living)
    attacker_win_rate = (tick_attacker_wins / tick_total_combats) if tick_total_combats > 0 else 0.0
    para_photo_ratio = (tick_parasitism_income / tick_total_gain) if tick_total_gain > 0 else 0.0

    return {
        "richness": richness,
        "shannon": shannon,
        "species_evenness": species_evenness,
        "age_structure": age_structure,
        "energy_gini": gini,
        "territory_stability": territory_stability,
        "trait_investment": trait_investment,
        "trait_corr": corr,
        "attacker_win_rate": attacker_win_rate,
        "para_photo_ratio": para_photo_ratio,
    }

             
#visualisations
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def setup_visualization():
    global main_window, grid_window, canvas, grid_canvas, im, e_im, biome_im
    global grid_ax, e_map_ax, biome_ax, obs_ax, corr_ax, energy_ax, trait_ax, age_ax
    global energy_line, photo_line, defense_line, invasive_line, parasite_line, trait_invest_line
    global gini_line, stability_line, attacker_line, para_photo_line, corr_im
    global age_lines, stats_text, debug_text
    # main window for graphs
    main_window = tk.Tk()
    main_window.title("Statistics")
    # main_window.state('zoomed')
    
    # separate window just for grid
    grid_window = tk.Toplevel(main_window)
    grid_window.title("World Grid")
    # grid_window.state('zoomed')
    
    # grid figure in its own window
    grid_fig = Figure(figsize=(15, 5))
    grid_ax = grid_fig.add_subplot(131)
    e_map_ax = grid_fig.add_subplot(132)
    biome_ax = grid_fig.add_subplot(133)
    grid_ax.axis('off')
    e_map_ax.axis('off')
    biome_ax.axis('off')
    grid_ax.set_title("Organisms")
    e_map_ax.set_title("Energy Map")
    biome_ax.set_title("Ecological Guilds")
    world_array = np.zeros((Width, Width, 3))
    e_map_arr = np.zeros((Width, Width))
    biome_arr = np.zeros((Width, Width, 3))
    im = grid_ax.imshow(world_array, interpolation='nearest')
    e_im = e_map_ax.imshow(e_map_arr, vmin=0, vmax=1, cmap='magma', interpolation='nearest')
    biome_im = biome_ax.imshow(biome_arr, interpolation='nearest')

    # Guild legend (on-screen)
    biome_ax.text(0.02, 0.98, "Guild Legend", transform=biome_ax.transAxes,
                  fontsize=9, fontweight='bold', va='top')
    biome_ax.text(0.02, 0.90, "Plant", transform=biome_ax.transAxes, color=(0.20, 0.85, 0.25), fontsize=8)
    biome_ax.text(0.02, 0.84, "Parasite", transform=biome_ax.transAxes, color=(0.88, 0.20, 0.80), fontsize=8)
    biome_ax.text(0.02, 0.78, "Invasive", transform=biome_ax.transAxes, color=(0.95, 0.45, 0.10), fontsize=8)
    biome_ax.text(0.02, 0.72, "Mixed", transform=biome_ax.transAxes, color=(0.20, 0.80, 0.95), fontsize=8)
    biome_ax.text(0.02, 0.64, "Hue tint = lineage", transform=biome_ax.transAxes, fontsize=7)

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
    
    # Observation trends
    obs_ax = fig.add_subplot(2, 3, 1)
    obs_ax.set_title("Observation Trends")
    obs_ax.set_xlabel("Tick")
    obs_ax.set_ylabel("Value")
    gini_line, = obs_ax.plot([], [], color='black', label='Energy Gini')
    stability_line, = obs_ax.plot([], [], color='teal', label='Territory Stability')
    attacker_line, = obs_ax.plot([], [], color='red', label='Attacker Win Rate')
    para_photo_line, = obs_ax.plot([], [], color='magenta', label='Parasitism/Photosynthesis')
    obs_ax.legend(fontsize=8)

    # Trait correlation heatmap
    corr_ax = fig.add_subplot(2, 3, 2)
    corr_ax.set_title("Trait Correlation Matrix")
    corr_im = corr_ax.imshow(np.eye(4), vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest')
    corr_ax.set_xticks([0, 1, 2, 3], labels=['P', 'D', 'I', 'Pa'])
    corr_ax.set_yticks([0, 1, 2, 3], labels=['P', 'D', 'I', 'Pa'])

    # Total energy over time
    energy_ax = fig.add_subplot(2, 3, 4)
    energy_ax.set_title("Total Energy Over Time")
    if history_total_energy:
        energy_ax.set_ylim(0, max(history_total_energy) * 1.1)  # 10% headroom
    energy_ax.set_xlabel("Tick")
    energy_ax.set_ylabel("Total Energy")
    energy_line, = energy_ax.plot([], [], 'r-')
    
    # Average traits over time
    trait_ax = fig.add_subplot(2, 3, 5)
    trait_ax.set_title("Average Traits Over Time")
    trait_ax.set_xlabel("Tick")
    trait_ax.set_ylabel("Trait Value")

    # Stats and debug text
    stats_ax = fig.add_subplot(2, 3, 3)
    stats_ax.axis('off')  # Hide axes for text display
    stats_text = stats_ax.text(0.05, 0.95, '', transform=stats_ax.transAxes,
                            fontsize=7, verticalalignment='top', family='monospace')

    age_ax = fig.add_subplot(2, 3, 6)
    age_ax.set_title("Age Structure Distribution")
    age_ax.set_xlabel("Tick")
    age_ax.set_ylabel("Fraction")

    debug_ax = stats_ax
    debug_ax.axis('off')
    debug_text = debug_ax.text(0.05, 0.05, '', transform=debug_ax.transAxes,
                            fontsize=8, verticalalignment='top', family='monospace')
    
    photo_line, = trait_ax.plot([], [], 'g-', label='Photosynthesis')
    defense_line, = trait_ax.plot([], [], 'b-', label='Defense')
    invasive_line, = trait_ax.plot([], [], 'r-', label='Invasiveness')
    parasite_line, = trait_ax.plot([], [], 'm-', label='Parasitism')
    trait_invest_line, = trait_ax.plot([], [], color='orange', label='Trait Investment')
    trait_ax.legend()

    age_lines = []
    age_colors = ['#4daf4a', '#377eb8', '#984ea3', '#ff7f00', '#a65628']
    for i, (lo, hi) in enumerate(age_bins):
        label = f"{lo}-{hi-1}" if hi is not None else f"{lo}+"
        line, = age_ax.plot([], [], color=age_colors[i % len(age_colors)], label=label)
        age_lines.append(line)
    age_ax.legend(fontsize=8)
    
    obs_ax.grid(True, alpha=0.3)
    energy_ax.grid(True, alpha=0.3)
    trait_ax.grid(True, alpha=0.3)
    age_ax.grid(True, alpha=0.3)

    fig.tight_layout()  # automatically adjust spacing
    
    canvas = FigureCanvasTkAgg(fig, master=main_window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # fill window

    # run archive/replay controls
    main_window.bind("<KeyPress-s>", on_save_run)
    main_window.bind("<KeyPress-r>", on_replay_latest)
    main_window.bind("<KeyPress-l>", on_exit_replay)
    main_window.bind("<KeyPress-o>", on_load_latest_saved_run)
    main_window.bind("<space>", on_toggle_replay_pause)
    main_window.bind("<Left>", on_replay_prev_frame)
    main_window.bind("<Right>", on_replay_next_frame)

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


def get_species_rgb(species_id, age, max_age):
    color_data = species_color[species_id]
    hue = color_data['base_hue']  # Use base_hue for consistent lineage color
    saturation = 0.8
    brightness = min(1, age/max_age) + 0.3  # Age controls brightness
    
    rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return rgb


def build_spatial_layers():
    world_array = np.zeros((Width, Width, 3), dtype=np.float32)
    energy_map_arr = np.zeros((Width, Width), dtype=np.float32)
    biome_map_arr = np.zeros((Width, Width, 3), dtype=np.float32)

    guild_base_colors = {
        "plant": (0.20, 0.85, 0.25),
        "parasite": (0.88, 0.20, 0.80),
        "invasive": (0.95, 0.45, 0.10),
        "mixed": (0.20, 0.80, 0.95),
    }

    for i in range(len(Alive)):
        if not Alive[i]:
            continue
        x, y = convert_to2D(tile_of[i])
        world_array[y][x] = get_species_rgb(species_id[i], Age[i], max_age)

        guild = classify_guild(i)
        base = guild_base_colors[guild]
        lineage_hue = species_color[species_id[i]]['base_hue']
        lineage_rgb = colorsys.hsv_to_rgb(lineage_hue, 0.55, 0.95)
        biome_map_arr[y][x] = (
            0.55 * base[0] + 0.45 * lineage_rgb[0],
            0.55 * base[1] + 0.45 * lineage_rgb[1],
            0.55 * base[2] + 0.45 * lineage_rgb[2],
        )

    for i in range(len(Energy_map)):
        x, y = convert_to2D(i)
        denom = (base_tile_energy * 1.05) if base_tile_energy > 0 else 1.0
        energy_map_arr[y][x] = Energy_map[i] / denom

    world_array = np.clip(world_array, 0, 1)
    energy_map_arr = np.clip(energy_map_arr, 0, 1)
    biome_map_arr = np.clip(biome_map_arr, 0, 1)
    return world_array, energy_map_arr, biome_map_arr


def reset_run_recording():
    global current_run_frames, current_capture_step
    current_run_frames = []
    current_capture_step = 0


def will_capture_frame_on_this_tick():
    return ((current_capture_step + 1) % FRAME_CAPTURE_STRIDE) == 0


def capture_run_frame(world_array=None, energy_map_arr=None, biome_map_arr=None):
    global current_capture_step, current_run_frames
    current_capture_step += 1
    if current_capture_step % FRAME_CAPTURE_STRIDE != 0:
        return False

    if world_array is None or energy_map_arr is None or biome_map_arr is None:
        world_array, energy_map_arr, biome_map_arr = build_spatial_layers()

    frame = {
        "world": (world_array * 255).astype(np.uint8),
        "energy": energy_map_arr.astype(np.float16),
        "biome": (biome_map_arr * 255).astype(np.uint8),
        "hist_idx": max(0, len(history_population) - 1),
        "stats_text": generate_stats_text(),
        "debug_text": generate_debug_text(),
    }
    current_run_frames.append(frame)
    if len(current_run_frames) > MAX_REPLAY_FRAMES:
        current_run_frames.pop(0)
    return True


def save_current_run(label=None):
    if not current_run_frames:
        return None

    name = label or f"run_{len(run_archive) + 1}"
    run_archive.append({
        "name": name,
        "saved_tick": len(history_population),
        "frames": [f for f in current_run_frames],
        "histories": {
            "population": history_population[:],
            "total_energy": history_total_energy[:],
            "photo": history_avg_photosynthesis[:],
            "defense": history_avg_defense[:],
            "invasive": history_avg_invasiveness[:],
            "parasite": history_avg_parasitism[:],
            "trait_invest": history_trait_investment[:],
            "energy_gini": history_energy_gini[:],
            "territory_stability": history_territory_stability[:],
            "attacker_win_rate": history_attacker_win_rate[:],
            "para_photo_ratio": history_para_photo_ratio[:],
            "age_structure": history_age_structure[:],
            "trait_corr": history_trait_corr[:],
            "species_richness": history_species_richness[:],
            "shannon": history_shannon_diversity[:],
            "evenness": history_evenness[:],
        }
    })
    if len(run_archive) > MAX_ARCHIVED_RUNS:
        run_archive.pop(0)
    return name


def save_run_to_disk(run_data, filename=None):
    os.makedirs(runs_dir, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fungi_run_{timestamp}.pkl"
    path = os.path.join(runs_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(run_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def list_saved_run_paths():
    if not os.path.isdir(runs_dir):
        return []
    files = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) if f.endswith(".pkl")]
    files.sort()
    return files


def load_run_from_disk(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_latest_run_from_disk():
    paths = list_saved_run_paths()
    if not paths:
        return None, None
    path = paths[-1]
    return load_run_from_disk(path), path


def estimate_memory_sinks():
    # Approximate live memory usage focused on dominant structures.
    details = {}
    list_targets = {
        "Alive": Alive,
        "Energy": Energy,
        "tile_of": tile_of,
        "Age": Age,
        "species_id": species_id,
        "invasiveness": invasiveness,
        "mutation_rate": mutation_rate,
        "home_court_potency": home_court_potency,
        "basalmetabolicrate": basalmetabolicrate,
        "offspring_energy_fraction": offspring_energy_fraction,
        "photosynthetic_ratio": photosynthetic_ratio,
        "spread_threshold": spread_threshold,
        "parasitism_rate": parasitism_rate,
        "World": World,
        "Energy_map": Energy_map,
        "history_population": history_population,
        "history_total_energy": history_total_energy,
        "history_trait_corr": history_trait_corr,
    }
    for name, arr in list_targets.items():
        details[name] = sys.getsizeof(arr) + (len(arr) * 8)

    frame_bytes = 0
    for frame in current_run_frames:
        frame_bytes += frame["world"].nbytes + frame["energy"].nbytes + frame["biome"].nbytes
    details["current_run_frames_arrays"] = frame_bytes
    details["run_archive_container"] = sys.getsizeof(run_archive)
    details["species_color_dict"] = sys.getsizeof(species_color)
    details["parent_registry_dict"] = sys.getsizeof(parent_registry)
    return details


def summarize_timing_hotspots(tick_profiles):
    phase_keys = [
        "instance",
        "spread",
        "energy_map",
        "refresh_best",
        "collect_stats",
        "build_layers",
        "capture_frame",
        "loop_total",
    ]
    out = {}
    if not tick_profiles:
        return out

    for key in phase_keys:
        values = [p["timing_ms"][key] for p in tick_profiles]
        out[key] = {
            "avg_ms": sum(values) / len(values),
            "max_ms": max(values),
            "p95_ms": sorted(values)[int(0.95 * (len(values) - 1))],
        }
    return out


def summarize_interesting_ticks(tick_profiles):
    if not tick_profiles:
        return {}

    by_time = max(tick_profiles, key=lambda p: p["timing_ms"]["loop_total"])
    by_gini = max(tick_profiles, key=lambda p: p["energy_gini"])
    by_shannon = max(tick_profiles, key=lambda p: p["shannon"])
    by_para_ratio = max(tick_profiles, key=lambda p: p["para_photo_ratio"])
    by_population_crash = min(tick_profiles, key=lambda p: p["population"])

    return {
        "slowest_tick": by_time,
        "max_inequality_tick": by_gini,
        "max_diversity_tick": by_shannon,
        "max_parasitism_pressure_tick": by_para_ratio,
        "lowest_population_tick": by_population_crash,
    }


def top_tracemalloc_allocations(snapshot, limit=12):
    if snapshot is None:
        return []
    stats = snapshot.statistics("lineno")
    rows = []
    for stat in stats[:limit]:
        rows.append({
            "location": str(stat.traceback[0]),
            "size_mb": round(stat.size / (1024 * 1024), 4),
            "count": stat.count,
        })
    return rows


def export_ai_analysis_bundle(run_name, saved_run_path, tick_profiles, snapshot=None):
    os.makedirs(analysis_dir, exist_ok=True)

    tick_profile_path = os.path.join(analysis_dir, f"{run_name}_tick_profile.jsonl")
    with open(tick_profile_path, "w", encoding="utf-8") as f:
        for row in tick_profiles:
            f.write(json.dumps(row) + "\n")

    memory_details = estimate_memory_sinks()
    ranked_memory = sorted(memory_details.items(), key=lambda kv: kv[1], reverse=True)
    ranked_memory = [
        {"name": name, "bytes": size, "mb": round(size / (1024 * 1024), 4)}
        for name, size in ranked_memory
    ]

    hotspot_summary = summarize_timing_hotspots(tick_profiles)
    interesting_ticks = summarize_interesting_ticks(tick_profiles)
    expensive_spots = sorted(
        hotspot_summary.items(),
        key=lambda kv: kv[1]["avg_ms"],
        reverse=True
    )

    analysis_summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": os.path.abspath(__file__),
        "saved_run_path": os.path.abspath(saved_run_path),
        "tick_profile_path": os.path.abspath(tick_profile_path),
        "environment_preset": ENVIRONMENT_PRESET,
        "config": {
            "Width": Width,
            "run_ticks_requested": run_ticks,
            "run_ticks_executed": len(tick_profiles),
            "FRAME_CAPTURE_STRIDE": FRAME_CAPTURE_STRIDE,
            "MAX_REPLAY_FRAMES": MAX_REPLAY_FRAMES,
            "sun_intensity": sun_intensity,
            "solar_efficiency": solar_efficiency,
            "triat_costmultiplier": triat_costmultiplier,
            "invasiveness_bonus_multipliar": invasiveness_bonus_multipliar,
            "parasitism_flat_drain": parasitism_flat_drain,
            "aggresion_reward": aggresion_reward,
            "defense_cost": defense_cost,
            "tau": tau,
            "alpha": alpha,
        },
        "preset_options_for_symbiosis": ENVIRONMENT_PRESETS,
        "interesting_ticks": interesting_ticks,
        "timing_hotspots_ms": hotspot_summary,
        "expensive_spots_ranked": [
            {"phase": phase, **vals} for phase, vals in expensive_spots
        ],
        "memory_sinks_ranked": ranked_memory,
        "top_allocations_tracemalloc": top_tracemalloc_allocations(snapshot),
        "notes": [
            "tick_profile.jsonl contains one row per tick for AI analysis.",
            "slowest_tick and memory_sinks_ranked point to optimization targets.",
            "source_file and code snapshot identify exact code under analysis.",
        ],
    }

    summary_path = os.path.join(analysis_dir, f"{run_name}_analysis.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)

    code_snapshot_path = None
    if EXPORT_CODE_SNAPSHOT:
        code_snapshot_path = os.path.join(analysis_dir, f"{run_name}_code_snapshot.py")
        with open(__file__, "r", encoding="utf-8") as src:
            code_snapshot = src.read()
        with open(code_snapshot_path, "w", encoding="utf-8") as dst:
            dst.write(code_snapshot)

    return {
        "summary_path": summary_path,
        "tick_profile_path": tick_profile_path,
        "code_snapshot_path": code_snapshot_path,
    }


def start_replay(run_index=-1):
    global replay_mode, replay_run_index, replay_frame_index, replay_playing
    if not run_archive:
        return False
    if run_index < 0:
        run_index = len(run_archive) + run_index
    if run_index < 0 or run_index >= len(run_archive):
        return False
    replay_mode = True
    replay_run_index = run_index
    replay_frame_index = 0
    replay_playing = True
    return True


def stop_replay():
    global replay_mode
    replay_mode = False


def render_replay_frame():
    global replay_frame_index
    if not run_archive:
        return

    run = run_archive[replay_run_index]
    frames = run["frames"]
    if not frames:
        return

    frame = frames[replay_frame_index]
    if replay_playing:
        replay_frame_index = (replay_frame_index + 1) % len(frames)

    im.set_data(frame["world"])
    e_im.set_data(frame["energy"].astype(np.float32))
    biome_im.set_data(frame["biome"])

    hist = run["histories"]
    hi = frame["hist_idx"]
    x, y = downsample(hist["total_energy"][:hi + 1]); energy_line.set_data(x, y)
    x, y = downsample(hist["photo"][:hi + 1]); photo_line.set_data(x, y)
    x, y = downsample(hist["defense"][:hi + 1]); defense_line.set_data(x, y)
    x, y = downsample(hist["invasive"][:hi + 1]); invasive_line.set_data(x, y)
    x, y = downsample(hist["parasite"][:hi + 1]); parasite_line.set_data(x, y)
    x, y = downsample(hist["trait_invest"][:hi + 1]); trait_invest_line.set_data(x, y)
    x, y = downsample(hist["energy_gini"][:hi + 1]); gini_line.set_data(x, y)
    x, y = downsample(hist["territory_stability"][:hi + 1]); stability_line.set_data(x, y)
    x, y = downsample(hist["attacker_win_rate"][:hi + 1]); attacker_line.set_data(x, y)
    x, y = downsample(hist["para_photo_ratio"][:hi + 1]); para_photo_line.set_data(x, y)
    corr_frames = hist.get("trait_corr", [])
    if corr_frames:
        corr_im.set_data(corr_frames[min(hi, len(corr_frames) - 1)])
    age_struct_series = hist.get("age_structure", [])
    for idx, line in enumerate(age_lines):
        vals = [row[idx] for row in age_struct_series[:hi + 1]] if age_struct_series else []
        x, y = downsample(vals)
        line.set_data(x, y)

    stats_text.set_text(frame["stats_text"])
    debug_text.set_text(
        frame["debug_text"]
        + f"\nPopulation: {hist['population'][min(hi, len(hist['population'])-1)]}"
        + f"\n\n=== REPLAY ===\nRun: {run['name']}\nFrame: {replay_frame_index + 1}/{len(frames)}"
    )

    obs_ax.relim(); obs_ax.autoscale_view()
    energy_ax.relim(); energy_ax.autoscale_view()
    trait_ax.relim(); trait_ax.autoscale_view()
    age_ax.relim(); age_ax.autoscale_view()
    grid_canvas.draw()
    canvas.draw()


def on_save_run(event=None):
    if not run_archive:
        name = save_current_run()
        if not name:
            print("[run save skipped] no frames captured yet")
            return
        run_data = run_archive[-1]
    else:
        run_data = run_archive[replay_run_index if replay_mode else -1]
    path = save_run_to_disk(run_data)
    print(f"[run saved to disk] {path}")


def on_replay_latest(event=None):
    if start_replay(-1):
        print(f"[replay] started {run_archive[replay_run_index]['name']}")
    else:
        print("[replay] no saved runs")


def on_load_latest_saved_run(event=None):
    run_data, path = load_latest_run_from_disk()
    if run_data is None:
        print("[load] no saved run files found")
        return
    run_archive.append(run_data)
    if len(run_archive) > MAX_ARCHIVED_RUNS:
        run_archive.pop(0)
    start_replay(-1)
    print(f"[load] loaded {path}")


def start_replay_from_disk(path=None):
    if path is None:
        run_data, loaded_path = load_latest_run_from_disk()
    else:
        run_data = load_run_from_disk(path)
        loaded_path = path

    if run_data is None:
        print("[startup] no saved runs found on disk")
        return False

    setup_visualization()
    run_archive.append(run_data)
    if len(run_archive) > MAX_ARCHIVED_RUNS:
        run_archive.pop(0)
    start_replay(-1)
    update_visualization()
    print(f"[startup] replaying saved run: {loaded_path}")
    return True


def on_exit_replay(event=None):
    stop_replay()
    print("[replay] stopped (no live simulation after fixed run)")


def on_toggle_replay_pause(event=None):
    global replay_playing
    if replay_mode:
        replay_playing = not replay_playing


def on_replay_prev_frame(event=None):
    global replay_frame_index, replay_playing
    if replay_mode and run_archive:
        replay_playing = False
        frame_count = len(run_archive[replay_run_index]["frames"])
        replay_frame_index = (replay_frame_index - 1) % frame_count


def on_replay_next_frame(event=None):
    global replay_frame_index, replay_playing
    if replay_mode and run_archive:
        replay_playing = False
        frame_count = len(run_archive[replay_run_index]["frames"])
        replay_frame_index = (replay_frame_index + 1) % frame_count


def run_headless_simulation(total_ticks):
    global previous_tile_species
    reset_run_recording()
    previous_tile_species = get_current_tile_species()
    tick_profiles = []
    snapshot = None

    if ENABLE_AI_PROFILING:
        tracemalloc.start(10)

    for t in range(total_ticks):
        tick()
        stats_start = time.perf_counter()
        collect_statistics()
        collect_stats_ms = (time.perf_counter() - stats_start) * 1000

        build_layers_ms = 0.0
        capture_frame_ms = 0.0
        if will_capture_frame_on_this_tick():
            build_start = time.perf_counter()
            world_array, energy_map_arr, biome_map_arr = build_spatial_layers()
            build_layers_ms = (time.perf_counter() - build_start) * 1000

            cap_start = time.perf_counter()
            capture_run_frame(world_array, energy_map_arr, biome_map_arr)
            capture_frame_ms = (time.perf_counter() - cap_start) * 1000
        else:
            capture_run_frame()

        if ENABLE_AI_PROFILING:
            tick_profiles.append({
                "tick": int(t + 1),
                "population": int(history_population[-1]),
                "total_energy": float(history_total_energy[-1]),
                "richness": int(history_species_richness[-1]),
                "shannon": float(history_shannon_diversity[-1]),
                "evenness": float(history_evenness[-1]),
                "energy_gini": float(history_energy_gini[-1]),
                "territory_stability": float(history_territory_stability[-1]),
                "attacker_win_rate": float(history_attacker_win_rate[-1]),
                "para_photo_ratio": float(history_para_photo_ratio[-1]),
                "timing_ms": {
                    "instance": float(tick_phase_instance_ms),
                    "spread": float(tick_phase_spread_ms),
                    "energy_map": float(tick_phase_energymap_ms),
                    "refresh_best": float(tick_phase_refresh_ms),
                    "collect_stats": float(collect_stats_ms),
                    "build_layers": float(build_layers_ms),
                    "capture_frame": float(capture_frame_ms),
                    "loop_total": float(
                        tick_phase_total_ms
                        + collect_stats_ms
                        + build_layers_ms
                        + capture_frame_ms
                    ),
                }
            })

    if ENABLE_AI_PROFILING and tracemalloc.is_tracing():
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

    run_name = f"run_{total_ticks}ticks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_current_run(run_name)
    run_data = run_archive[-1]
    saved_path = save_run_to_disk(run_data, f"{run_name}.pkl")
    analysis_paths = export_ai_analysis_bundle(run_name, saved_path, tick_profiles, snapshot)
    return {
        "saved_run_path": saved_path,
        "analysis_summary_path": analysis_paths["summary_path"],
        "tick_profile_path": analysis_paths["tick_profile_path"],
        "code_snapshot_path": analysis_paths["code_snapshot_path"],
    }


def update_visualization():
    if replay_mode:
        render_replay_frame()
    main_window.after(replay_interval_ms, update_visualization)



def collect_statistics():
    global tick_total_gain, tick_total_bmr, tick_metabolism_count, tick_combat_income
    living = [i for i in range(len(Alive)) if Alive[i]]
    history_population.append(len(living))
    history_total_energy.append(sum(Energy[i] for i in living))
    if living:  # avoid division by zero
        history_avg_photosynthesis.append(sum(photosynthetic_ratio[i] for i in living) / len(living))
        history_avg_defense.append(sum(home_court_potency[i] for i in living) / len(living))
        history_avg_age.append(sum(Age[i] for i in living) / len(living))
        history_avg_invasiveness.append(sum(invasiveness[i] for i in living) / len(living))
        history_avg_parasitism.append(sum(parasitism_rate[i] for i in living) / len(living))
    else:
        history_avg_photosynthesis.append(0)
        history_avg_defense.append(0)
        history_avg_age.append(0)
        history_avg_invasiveness.append(0)
        history_avg_parasitism.append(0)

    if tick_metabolism_count:
        history_avg_gain.append(tick_total_gain / tick_metabolism_count)
        history_avg_bmr.append(tick_total_bmr / tick_metabolism_count)
    else:
        history_avg_gain.append(0)
        history_avg_bmr.append(0)
    history_combat_income.append(tick_combat_income)

    obs = compute_observations(living)
    history_species_richness.append(obs["richness"])
    history_shannon_diversity.append(obs["shannon"])
    history_evenness.append(obs["species_evenness"])
    history_age_structure.append(obs["age_structure"])
    history_energy_gini.append(obs["energy_gini"])
    history_territory_stability.append(obs["territory_stability"])
    history_trait_investment.append(obs["trait_investment"])
    history_trait_corr.append(obs["trait_corr"])
    history_attacker_win_rate.append(obs["attacker_win_rate"])
    history_para_photo_ratio.append(obs["para_photo_ratio"])

    if len(history_population) % CLEANUP_INTERVALS == 0:
        trim_history()

#optimization to reduce amount of ticks being tracked
def trim_history():
    global history_population, history_total_energy, history_avg_age, history_avg_photosynthesis
    global history_avg_defense, history_avg_invasiveness, history_avg_parasitism
    global history_avg_gain, history_avg_bmr, history_combat_income
    global history_species_richness, history_shannon_diversity, history_evenness
    global history_age_structure, history_energy_gini, history_territory_stability
    global history_trait_investment, history_trait_corr, history_attacker_win_rate, history_para_photo_ratio

    if len(history_population) > MAX_HISTORY:
        trim = len(history_population) - MAX_HISTORY

        history_population = history_population[trim:]
        history_total_energy = history_total_energy[trim:]
        history_avg_photosynthesis = history_avg_photosynthesis[trim:]
        history_avg_defense = history_avg_defense[trim:]
        history_avg_age = history_avg_age[trim:]
        history_avg_invasiveness = history_avg_invasiveness[trim:]
        history_avg_parasitism = history_avg_parasitism[trim:]
        history_avg_gain = history_avg_gain[trim:]
        history_avg_bmr = history_avg_bmr[trim:]
        history_combat_income = history_combat_income[trim:]
        history_species_richness = history_species_richness[trim:]
        history_shannon_diversity = history_shannon_diversity[trim:]
        history_evenness = history_evenness[trim:]
        history_age_structure = history_age_structure[trim:]
        history_energy_gini = history_energy_gini[trim:]
        history_territory_stability = history_territory_stability[trim:]
        history_trait_investment = history_trait_investment[trim:]
        history_trait_corr = history_trait_corr[trim:]
        history_attacker_win_rate = history_attacker_win_rate[trim:]
        history_para_photo_ratio = history_para_photo_ratio[trim:]


def generate_stats_text():
    text = "=== CURRENT BEST ===\n"
    richness = history_species_richness[-1] if history_species_richness else 0
    shannon = history_shannon_diversity[-1] if history_shannon_diversity else 0
    evenness = history_evenness[-1] if history_evenness else 0
    gini = history_energy_gini[-1] if history_energy_gini else 0
    stability = history_territory_stability[-1] if history_territory_stability else 0
    attacker_wr = history_attacker_win_rate[-1] if history_attacker_win_rate else 0
    para_photo_ratio = history_para_photo_ratio[-1] if history_para_photo_ratio else 0
    
    if current_best['energy_index'] is not None:
        i = current_best['energy_index']
        text += f"Highest Energy: {Energy[i]:.0f}\n"
        text += f"  Age: {Age[i]}, Species: {species_id[i]}\n"
        text += f"  P:{photosynthetic_ratio[i]:.2f} D:{home_court_potency[i]:.2f} I:{invasiveness[i]:.2f} Pa:{parasitism_rate[i]:.2f} S:{spread_threshold[i]:.0f}\n\n"
    
    if current_best['age_index'] is not None:
        i = current_best['age_index']
        text += f"Oldest: {Age[i]} ticks\n"
        text += f"  Energy: {Energy[i]:.0f}, Species: {species_id[i]}\n"
        text += f"  P:{photosynthetic_ratio[i]:.2f} D:{home_court_potency[i]:.2f} I:{invasiveness[i]:.2f} Pa:{parasitism_rate[i]:.2f} S:{spread_threshold[i]:.0f}\n\n"
    
    text += "=== ALL-TIME RECORDS ===\n"
    if best_organism_ever['energy_record_holder']:
        rec = best_organism_ever['energy_record_holder']
        text += f"Energy Record: {rec['energy']:.0f}\n"
        text += f"  Age: {rec['age']}, Species: {rec['species_id']}\n"
        text += f"  P:{rec['photosynthesis']:.2f} D:{rec['defense']:.2f} I:{rec['invasiveness']:.2f} Pa:{rec['parasitism']:.2f} S:{rec['spread_threshold']:.0f}\n\n"
    
    if best_organism_ever['age_record_holder']:
        rec = best_organism_ever['age_record_holder']
        text += f"Age Record: {rec['age']} ticks\n"
        text += f"  Energy: {rec['energy']:.0f}, Species: {rec['species_id']}\n"
        text += f"  P:{rec['photosynthesis']:.2f} D:{rec['defense']:.2f} I:{rec['invasiveness']:.2f} Pa:{rec['parasitism']:.2f} S:{rec['spread_threshold']:.0f}"

    text += "\n\n=== ECOSYSTEM HEALTH ===\n"
    text += f"Richness: {richness} species\n"
    text += f"Shannon Diversity: {shannon:.2f}\n"
    text += f"Species Evenness: {evenness:.2f}\n"
    text += f"Energy Gini: {gini:.2f}\n"
    text += f"Territory Stability: {stability:.2f}\n"
    text += f"Attacker Win Rate: {attacker_wr:.2f}\n"
    text += f"Parasitism/Photosynthesis: {para_photo_ratio:.2f}\n"
    
    return text

def generate_debug_text():
    avg_gain = history_avg_gain[-1] if history_avg_gain else 0
    avg_bmr = history_avg_bmr[-1] if history_avg_bmr else 0
    combat_income = history_combat_income[-1] if history_combat_income else 0
    richness = history_species_richness[-1] if history_species_richness else 0
    shannon = history_shannon_diversity[-1] if history_shannon_diversity else 0
    evenness = history_evenness[-1] if history_evenness else 0
    gini = history_energy_gini[-1] if history_energy_gini else 0
    stability = history_territory_stability[-1] if history_territory_stability else 0
    invest = history_trait_investment[-1] if history_trait_investment else 0
    attacker_wr = history_attacker_win_rate[-1] if history_attacker_win_rate else 0
    para_photo_ratio = history_para_photo_ratio[-1] if history_para_photo_ratio else 0
    age_struct = history_age_structure[-1] if history_age_structure else [0] * len(age_bins)

    text = "=== ENERGY FLOW (LAST TICK) ===\n"
    text += f"Avg Gain/Org: {avg_gain:.3f}\n"
    text += f"Avg BMR/Org:  {avg_bmr:.3f}\n"
    text += f"Combat Income: {combat_income:.3f}\n"
    text += f"Net (Gain-BMR): {(avg_gain - avg_bmr):.3f}\n\n"
    text += "=== OBSERVATIONS (LAST TICK) ===\n"
    text += f"Richness: {richness}  Shannon: {shannon:.2f}  Evenness: {evenness:.2f}\n"
    text += f"Energy Gini: {gini:.2f}  Territory Stability: {stability:.2f}\n"
    text += f"Trait Investment Avg: {invest:.2f}\n"
    text += f"Attacker Win Rate: {attacker_wr:.2f}\n"
    text += f"Parasitism/Photosynthesis: {para_photo_ratio:.2f}\n"
    age_labels = []
    for i, (lo, hi) in enumerate(age_bins):
        label = f"{lo}-{hi-1}" if hi is not None else f"{lo}+"
        age_labels.append(f"{label}:{age_struct[i]:.2f}")
    text += "Age Structure: " + " ".join(age_labels) + "\n"
    text += f"Runs Saved: {len(run_archive)}\n"
    text += "Controls: S save | O load latest file | R replay | L stop replay | Space pause | <- -> step"
    return text



def start():
    apply_environment_preset(ENVIRONMENT_PRESET)
    build_neighbor_cache()
    spatial_occupancy.clear()
    reset_energy_map()
    initialize(initial_organism_count)
    run_outputs = run_headless_simulation(run_ticks)
    print(f"[simulation complete] ticks={run_ticks}")
    print(f"[preset] {ENVIRONMENT_PRESET}")
    print(f"[saved] {run_outputs['saved_run_path']}")
    print(f"[analysis] {run_outputs['analysis_summary_path']}")
    print(f"[tick profile] {run_outputs['tick_profile_path']}")
    if run_outputs["code_snapshot_path"]:
        print(f"[code snapshot] {run_outputs['code_snapshot_path']}")

    setup_visualization()
    start_replay(-1)
    update_visualization()
    


#helper functions
def find_neighbour1D(tile :int):
    if not neighbor_cache or len(neighbor_cache) != (Width * Width):
        build_neighbor_cache()
    return neighbor_cache[tile]


def convert_to2D(tile : int):
    y, x = divmod(tile, Width)
    return (x, y)

def convert_to1D(coords : tuple):
    return(coords[1]*Width + coords[0])


def build_neighbor_cache():
    global neighbor_cache
    cache = [[] for _ in range(Width * Width)]
    for tile in range(Width * Width):
        if tile % Width != 0:
            cache[tile].append(tile - 1)
        if tile % Width != Width - 1:
            cache[tile].append(tile + 1)
        if tile - Width >= 0:
            cache[tile].append(tile - Width)
        if tile + Width < Width * Width:
            cache[tile].append(tile + Width)
    neighbor_cache = [tuple(nbrs) for nbrs in cache]

STARTUP_MODE = "simulate"  # "simulate" or "replay_latest"

if __name__ == "__main__":
    if STARTUP_MODE == "replay_latest":
        if start_replay_from_disk():
            main_window.mainloop()
    else:
        start()
        main_window.mainloop()
