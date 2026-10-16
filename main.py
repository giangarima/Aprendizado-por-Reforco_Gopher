import copy
import csv
import os
import random

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

# ---------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------
ENV_ID = "ALE/Gopher-v5"

TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ENTRADAS = 128
NUM_OCULTOS = 16
NUM_ACOES = 8
NUM_EPISODIOS_BASELINE = 20

TAXA_MUTACAO = 0.01
INTENSIDADE_MUTACAO = 0.05
NUM_ELITE = 2


def criar_ambiente(render_mode=None):
    return gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode=render_mode
    )


class Agente:
    def __init__(self):
        self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
        self.b1 = np.random.randn(NUM_OCULTOS)
        self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
        self.b2 = np.random.randn(NUM_ACOES)
        self.fitness = 0.0

    def agir(self, estado):
        estado = estado.astype(np.float32) / 255.0

        oculto = np.tanh(estado @ self.W1 + self.b1)
        saida = oculto @ self.W2 + self.b2

        return int(np.argmax(saida))


def avaliar(agente):
    env = criar_ambiente()
    estado, info = env.reset()

    recompensa_total = 0.0
    terminou = False

    while not terminou:
        acao = agente.agir(estado)
        estado, reward, terminated, truncated, info = env.step(acao)
        recompensa_total += reward
        terminou = terminated or truncated

    env.close()
    return recompensa_total


def avaliar_agente_aleatorio():
    env = criar_ambiente()
    resultados = []

    for episodio in range(NUM_EPISODIOS_BASELINE):
        env.reset(seed=episodio)
        env.action_space.seed(episodio)
        recompensa_total = 0.0
        terminou = False

        while not terminou:
            acao = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(acao)
            recompensa_total += reward
            terminou = terminated or truncated

        resultados.append({
            "episodio": episodio,
            "fitness": recompensa_total,
        })

    env.close()
    return resultados


def salvar_baseline_aleatorio():
    resultados = avaliar_agente_aleatorio()
    fitness = [resultado["fitness"] for resultado in resultados]

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/baseline_aleatorio.csv", "w", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["episodio", "fitness"])
        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"Episódios aleatórios avaliados: {NUM_EPISODIOS_BASELINE}")
    print(f"Fitness médio aleatório: {np.mean(fitness):.2f}")
    print(f"Melhor fitness aleatório: {np.max(fitness):.2f}")
    print("Baseline salvo em resultados/baseline_aleatorio.csv")


def criar_populacao():
    return [Agente() for _ in range(TAMANHO_POPULACAO)]


def selecionar(populacao):
    ordenada = sorted(populacao, key=lambda a: a.fitness, reverse=True)
    n_pais = max(NUM_ELITE, len(populacao) // 2)
    return ordenada[:n_pais]


def mutar(agente):
    for nome in ("W1", "b1", "W2", "b2"):
        param = getattr(agente, nome)
        mascara = np.random.rand(*param.shape) < TAXA_MUTACAO
        ruido = np.random.randn(*param.shape) * INTENSIDADE_MUTACAO
        param += mascara * ruido


def nova_geracao(populacao):
    ordenada = sorted(populacao, key=lambda a: a.fitness, reverse=True)
    pais = selecionar(populacao)

    nova = [copy.deepcopy(a) for a in ordenada[:NUM_ELITE]]

    while len(nova) < TAMANHO_POPULACAO:
        filho = copy.deepcopy(random.choice(pais))
        mutar(filho)
        filho.fitness = 0.0
        nova.append(filho)

    return nova


def visualizar(agente, max_frames=5000):
    env = criar_ambiente(render_mode="human")
    estado, info = env.reset()
    total = 0.0

    for _ in range(max_frames):
        acao = agente.agir(estado)
        estado, reward, terminated, truncated, info = env.step(acao)
        total += reward
        env.render()

        if terminated or truncated:
            break

    env.close()
    print(f"Fitness na demonstração: {total:.2f}")


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    salvar_baseline_aleatorio()

    populacao = criar_populacao()
    historico = []

    melhor_agente = None
    melhor_fitness = -float("inf")

    for geracao in range(NUM_GERACOES):

        for agente in populacao:
            agente.fitness = avaliar(agente)

            if agente.fitness > melhor_fitness:
                melhor_fitness = agente.fitness
                melhor_agente = copy.deepcopy(agente)

        valores = [agente.fitness for agente in populacao]

        melhor = max(valores)
        media = float(np.mean(valores))
        pior = min(valores)

        historico.append({
            "geracao": geracao,
            "melhor": melhor,
            "media": media,
            "pior": pior,
        })

        print(
            f"Geração {geracao:03d} | "
            f"Melhor: {melhor:.2f} | "
            f"Média: {media:.2f} | "
            f"Pior: {pior:.2f}"
        )

        populacao = nova_geracao(populacao)

    if historico:
        with open("resultados/fitness.csv", "w", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=["geracao", "melhor", "media", "pior"])
            escritor.writeheader()
            escritor.writerows(historico)

        geracoes = [h["geracao"] for h in historico]

        plt.figure(figsize=(8, 5))
        plt.plot(geracoes, [h["melhor"] for h in historico], label="Melhor fitness")
        plt.plot(geracoes, [h["media"] for h in historico], label="Fitness médio")
        plt.xlabel("Gerações")
        plt.ylabel("Fitness")
        plt.title("Evolução do agente no Gopher")
        plt.legend()
        plt.grid(True)
        plt.savefig("resultados/fitness.png", dpi=150)
        plt.close()

    if melhor_agente is not None:
        visualizar(melhor_agente)
        print(f"Fitness durante o treino: {melhor_fitness:.2f}")
    else:
        print("Nenhum agente válido foi encontrado durante o treino.")