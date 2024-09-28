import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms
import tkinter as tk

genes = {
    "Gene_A": 0.8,
    "Gene_B": 0.65,
    "Gene_C": 0.9,
    "Gene_D": 0.7,
    "Gene_E": 0.6,
}

def get_bangalore_weather():
    temperature = random.uniform(20, 35)
    humidity = random.uniform(40, 80)
    wind_speed = random.uniform(5, 20)
    air_pressure = random.uniform(1000, 1025)
    return temperature, humidity, wind_speed, air_pressure

def gene_combination_efficiency(individual):
    temperature, humidity, wind_speed, air_pressure = get_bangalore_weather()
    weather_factor = (1 - ((abs(temperature - 25) + abs(humidity - 60)) / 100)) * \
                     (1 - (wind_speed / 50)) * (air_pressure / 1013)
    efficiency = sum(genes[gene] for gene, selected in zip(genes.keys(), individual) if selected)
    efficiency *= weather_factor
    return (efficiency,)

def calculate_success_ratio(individual):
    base_success_rate = 80
    weather_factor = random.uniform(0.8, 1.0)
    gene_count_factor = 1 - (sum(individual) * 0.05)
    random_noise = random.uniform(0.9, 1.1)
    success_ratio = base_success_rate * weather_factor * gene_count_factor * random_noise
    return max(0, success_ratio)

def generate_advantages_disadvantages(individual):
    selected_genes = [gene for gene, selected in zip(genes.keys(), individual) if selected]
    
    advantages = []
    disadvantages = []
    key_points = []

    if "Gene_A" in selected_genes:
        advantages.append("High pollutant absorption capacity")
        disadvantages.append("May require additional engineering in unstable climates")
        key_points.append("Gene_A performs best under moderate temperature.")

    if "Gene_B" in selected_genes:
        advantages.append("Resistant to moderate humidity changes")
        disadvantages.append("Limited absorption under high wind speeds")
        key_points.append("Ensure controlled environments with lower wind speeds for optimal performance.")

    if "Gene_C" in selected_genes:
        advantages.append("Highly effective at pollutant absorption, even under high temperature")
        disadvantages.append("Sensitive to sharp humidity changes")
        key_points.append("Use Gene_C in relatively stable humid conditions.")

    if "Gene_D" in selected_genes:
        advantages.append("Good for urban areas with variable weather conditions")
        disadvantages.append("Lower efficiency in low-pressure environments")
        key_points.append("Monitor air pressure when deploying Gene_D in city environments.")

    if "Gene_E" in selected_genes:
        advantages.append("Efficient under low humidity")
        disadvantages.append("Loses effectiveness in high humidity environments")
        key_points.append("Deploy Gene_E in drier climates for best results.")

    return advantages, disadvantages, key_points

def show_results(selected_genes, efficiency, success_ratio, weather, advantages, disadvantages, key_points):
    result_window = tk.Tk()
    result_window.title("Gene Analysis Results")

    tk.Label(result_window, text="Selected Genes: " + ", ".join(selected_genes), font=("Arial", 14)).pack(pady=5)
    tk.Label(result_window, text=f"Efficiency: {efficiency:.2f}%", font=("Arial", 12)).pack(pady=5)
    tk.Label(result_window, text=f"Success Ratio: {success_ratio:.2f}%", font=("Arial", 12)).pack(pady=5)
    tk.Label(result_window, text=f"Weather: Temp={weather[0]:.2f}°C, Humidity={weather[1]:.2f}%, Wind={weather[2]:.2f} km/h, Pressure={weather[3]:.2f} hPa", font=("Arial", 12)).pack(pady=5)

    tk.Label(result_window, text="Advantages:", font=("Arial", 14)).pack(pady=5)
    for advantage in advantages:
        tk.Label(result_window, text=f"• {advantage}", font=("Arial", 12)).pack(anchor='w')

    tk.Label(result_window, text="\nDisadvantages:", font=("Arial", 14)).pack(pady=5)
    for disadvantage in disadvantages:
        tk.Label(result_window, text=f"• {disadvantage}", font=("Arial", 12)).pack(anchor='w')

    tk.Label(result_window, text="\nKey Points:", font=("Arial", 14)).pack(pady=5)
    for point in key_points:
        tk.Label(result_window, text=f"• {point}", font=("Arial", 12)).pack(anchor='w')

    tk.Button(result_window, text="Close", command=result_window.destroy).pack(pady=10)
    result_window.mainloop()

def run_genetic_algorithm():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(genes))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", gene_combination_efficiency)

    population_size = 50
    crossover_prob = 0.7
    mutation_prob = 0.2
    generations = 30

    population = toolbox.population(n=population_size)

    algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations, verbose=False)

    top_individuals = tools.selBest(population, k=3)

    for idx, individual in enumerate(top_individuals):
        efficiency = sum(genes[gene] for gene, selected in zip(genes.keys(), individual) if selected)
        success_ratio = calculate_success_ratio(individual)
        weather = get_bangalore_weather()
        selected_genes = [gene for gene, selected in zip(genes.keys(), individual) if selected]

        advantages, disadvantages, key_points = generate_advantages_disadvantages(individual)

        show_results(selected_genes, efficiency * 100, success_ratio, weather, advantages, disadvantages, key_points)

    plot_gene_success_ratio([sum(genes[gene] for gene, selected in zip(genes.keys(), ind) if selected) * 100 for ind in top_individuals], 
                            [calculate_success_ratio(ind) for ind in top_individuals])
    plot_weather_impact([get_bangalore_weather() for _ in range(3)])

def plot_gene_success_ratio(efficiency_results, success_ratios):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Result 1', 'Result 2', 'Result 3'], efficiency_results, color='blue', alpha=0.6, label="Efficiency")
    ax.bar(['Result 1', 'Result 2', 'Result 3'], success_ratios, color='green', alpha=0.6, label="Success Ratio")
    ax.set_xlabel("Gene Combinations")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Gene Combination Efficiency and Success Ratios")
    ax.legend()
    plt.show()

def plot_weather_impact(weather_results):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    temperatures = [weather[0] for weather in weather_results]
    humidities = [weather[1] for weather in weather_results]
    wind_speeds = [weather[2] for weather in weather_results]
    
    ax.scatter(temperatures, humidities, wind_speeds, color='red', s=100)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_zlabel("Wind Speed (km/h)")
    ax.set_title("Weather Impact on Gene Efficiency")
    plt.show()

if __name__ == "__main__":
    run_genetic_algorithm()
