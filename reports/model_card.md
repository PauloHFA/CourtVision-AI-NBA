# Model card — CourtVision AI

Status: **rascunho**. `performance_verified: false`.

Este arquivo descreve o contrato do modelo. Ele não afirma que um champion já foi treinado em dados reais nem que superou os baselines.

## Uso pretendido

Estimar a probabilidade de vitória do mandante em um confronto, usando apenas informação anterior ao corte `as_of`. A saída é educacional.

## Fora de uso

- Aconselhamento de apostas, odds ou garantia de resultado.
- Impacto causal de um jogador ausente.
- Feed substituto de estatísticas.

## Alvo

`y = 1` se o mandante vence um jogo finalizado da temporada regular. Não há empate no rótulo.

## Features

Versão `features-1`. Janelas de 5 e 10 jogos, mando, descanso, Elo pré-jogo e indicadores de imputação. Jogos do mesmo dia civil do corte ficam de fora.

## Baselines obrigatórios

- M0: probabilidade 0,5
- M1: taxa histórica de vitória do mandante, atualizada só com jogos anteriores ao corte
- M2: regressão logística regularizada
- M3: XGBoost, se o extra `ml` estiver instalado. A busca curta usa só o treino.

## Promoção

O challenger só vira champion se o Brier de validação for menor que o dos baselines, com a folga proposta em `material_brier_gain`. Esse gate não é um resultado já medido. A temporada de teste não entra na seleção. A calibração, se aplicada, usa só a fatia final da validação e também não promove o modelo. Se ninguém superar o baseline, o produto permanece transparente e não anuncia modelo superior.

## Limitações conhecidas

- O fixture do repositório é sintético.
- A data em que um placar se tornou publicamente conhecido não vem da fonte; resumos textuais usam atraso conservador de 1 dia.
- Playoffs ficam fora do recorte inicial.
