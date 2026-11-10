# CourtVision AI

Estudo educacional para estimar a probabilidade de vitória de um mandante e explicar a estimativa com evidências. Não é afiliado à NBA, à BALLDONTLIE ou a qualquer equipe. Não recomenda apostas e não garante resultado.

A especificação canônica está em `CourtVision_AI_Especificacao_Tecnica_Completa.pdf`. O texto extraído em `docs/especificacao_extraida.txt` serve só para busca.

## Estado desta base

Isto é o começo do roteiro, não o produto final.

Pronto agora:

- estrutura do repositório, licença MIT só para o código, Compose e CI
- schema relacional e migração Alembic
- cliente da fonte com paginação, retry e parada em 401/403
- importação direta do fixture sintético para banco relacional via CLI
- features pré-jogo com corte por dia e testes anti-vazamento
- previsão com baseline histórico, abstenção quando falta histórico e contrato de probabilidade
- API `/health`, `/teams`, `/games`, `/predictions`, `/assistant/ask`
- interface React com painel, comparador, evidências clicáveis e laboratório sintético
- treino que compara 50/50, taxa do mandante e logística, sem usar a temporada de teste na seleção
- challenger XGBoost quando o extra `ml` está instalado; a busca fica só no treino
- calibração sigmoid na fatia final da validação, sem usar esse ajuste para promover o modelo
- matriz curada de 80 perguntas do RAG balanceadas e avaliadas (protocolo e dataset em `evaluation/rag_questions_80.json`)
- módulo e CLI de indexação do RAG para as tabelas `documents` e `document_chunks`

Ainda não feito, de propósito:

- coleta real da API e snapshot histórico auditado
- champion treinado em dados reais e avaliação externa
- embeddings pgvector em produção e LLM local obrigatório
- simulação de jogador ausente

`performance_verified` permanece `false`.

## Como rodar

Requisitos: Python 3.12+, Node.js LTS e, para o banco, Docker.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
# Para incluir o challenger XGBoost: pip install -e ".[dev,ml]"
uvicorn apps.api.main:app --reload
```

Sem `DATABASE_URL`, a API sobe com o fixture sintético. Em outro terminal:

```powershell
cd apps\web
npm install
npm run dev
```

Abra `http://127.0.0.1:5173`. A data de corte de demonstração é 1º de abril de 2024, quando o fixture já tem histórico.

Postgres e Ollama locais:

```powershell
copy .env.example .env
docker compose up -d postgres ollama
alembic upgrade head
```

A senha `localdev` existe só para a máquina de desenvolvimento. A chave `BDL_API_KEY` fica no `.env`, nunca no frontend nem no Git.

Importar o fixture, indexar para RAG e gerar um relatório de qualidade:

```powershell
python -m courtvision.ingestion.cli --fixture data/fixtures/synthetic_season.json
# Para persistir no Postgres/SQLite:
# python -m courtvision.ingestion.cli --fixture data/fixtures/synthetic_season.json --database-url <DATABASE_URL>
python -m courtvision.rag.index --fixture data/fixtures/synthetic_season.json --database-url <DATABASE_URL>
python -m courtvision.evaluation.cli --fixture data/fixtures/synthetic_season.json
python -m courtvision.ml.train --config configs/train.yaml
```

O treino só promove um champion se o Brier de validação superar os baselines. No fixture sintético, isso pode não acontecer. Esse resultado negativo é válido.

## Decisões já registradas

Estão em `configs/decisions.yaml`. O corte temporal ignora jogos do mesmo dia. A população supervisionada inicial é temporada regular finalizada. Dados brutos da API não entram no repositório.

## Licença

MIT para o código original. A licença não relicencia dados, marcas ou conteúdo de terceiros.
