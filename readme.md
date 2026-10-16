# Evolução de um Agente Inteligente no Atari — Gopher

## Ambiente

- **Ambiente:** `ALE/Gopher-v5`
- **Observação:** RAM do Atari (vetor com 128 valores)
- **Ações:** 8 ações discretas:

| Índice | Ação |
|---:|---|
| 0 | NOOP |
| 1 | FIRE |
| 2 | UP |
| 3 | RIGHT |
| 4 | LEFT |
| 5 | UPFIRE |
| 6 | RIGHTFIRE |
| 7 | LEFTFIRE |

- **Fitness:** soma de todas as recompensas recebidas pelo agente durante um episódio.

## Como executar

```bash
pip install "gymnasium[atari]" numpy matplotlib
python main.py
```

## Seleção, elitismo e mutação

- **Seleção:** metade da população com maior fitness vira pai.
- **Elitismo:** os `NUM_ELITE` melhores são copiados sem mutação para a próxima geração.
- **Mutação:** cada peso tem uma probabilidade definida por `TAXA_MUTACAO` de ser alterado. Quando isso acontece, é acrescentada uma variação aleatória, cujo tamanho é controlado por `INTENSIDADE_MUTACAO`.

## Experimentos

| Parâmetro | A | B | C |
|---|---:|---:|---:|
| População | 20 | 20 | 20 |
| Gerações | 30 | 30 | 30 |
| Elite | 2 | 2 | 2 |
| Taxa de mutação | 1% | 10% | 40% |
| Intensidade | 0,05 | 0,20 | 0,50 |
| Melhor fitness | 1100 | 960 | **1820** |
| Média final | 244 | 334 | 317 |
| Pior final | 80 | 80 | 40 |

Os CSVs e gráficos dos experimentos A, B e C estão na pasta `resultados/`.

## Análise

A exploração vem da população inicial aleatória e das mutações. O experimento B teve a maior média final, com 334, mas os resultados ainda apresentaram variações entre as gerações. O experimento C teve o maior pico de fitness, com 1820, mas não apresentou convergência. A média continuou variando e o pior fitness voltou a zero várias vezes. O experimento A teve uma evolução mais lenta por conta da baixa mutação, alcançando o melhor fitness de 1100, mas terminou com a menor média final. Dessa forma, nenhum dos três experimentos apresentou uma convergência clara.

O agente evoluído superou o melhor resultado do agente aleatório em todos os experimentos:
- Melhor fitness do experimento A: 1100
- Melhor fitness do experimento B: 960
- Melhor fitness do experimento C: 1820
- Fitness médio aleatório: 327,00
- Melhor fitness aleatório: 860,00