# Dashboard (Streamlit)

## Ejecutar en servidor

1) Instala dependencias:
```
pip install -r requirements.txt
```

2) (Opcional) Define la ruta a la base DuckDB:
```
setx EXPERIMENTS_DB_PATH "C:\ruta\experiments.duckdb"
```

3) Lanza el dashboard:
```
streamlit run app.py
```

Si no se define `EXPERIMENTS_DB_PATH`, el dashboard usa:
`dashboard/data/experiments.duckdb`

---

## LLM Trace Viewer

El dashboard incluye un visor de trazas LLM que permite visualizar las respuestas de la API de OpenAI de forma estructurada.

### Características

- **Vista de Timeline**: Muestra cada paso del proceso (razonamiento, llamadas a herramientas, resultados, output final)
- **Filtros**: Filtra por tipo de paso (Reasoning, Tools, Output, Errors)
- **Búsqueda**: Busca texto dentro de la traza
- **Vista Raw JSON**: Mantiene acceso al JSON original sin modificar
- **Sin truncamiento**: Todo el contenido se muestra completo con scroll si es necesario

### Formatos Soportados

El parser soporta tres formatos de respuesta:

1. **OpenAI Responses API** (`object: "response"`)
   - Soporta reasoning, file_search, function_call, web_search
   - Maneja estados incompletos y errores

2. **OpenAI Chat Completions API** (`object: "chat.completion"`)
   - Soporta tool_calls y múltiples choices
   - Maneja finish_reason (stop, length, tool_calls)

3. **Custom RAG Format** (usado en este proyecto)
   - Campos: `razonamiento`, `codigos`, `conceptos`
   - Maneja errores en campo `error`

### Configuración

Variables de entorno para controlar el comportamiento:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `TRACE_VIEWER_ENABLED` | `1` | Habilita/deshabilita el visor de trazas |
| `TRACE_VIEWER_DEFAULT_TAB` | `trace` | Tab por defecto (`trace` o `json`) |
| `TRACE_VIEWER_MAX_JSON_SIZE` | `1000000` | Tamaño máximo de JSON antes de warning (bytes) |

### Arquitectura

```
dashboard/
├── app.py                 # Aplicación principal Streamlit
├── trace_parser.py        # Parser de respuestas → TraceModel
├── trace_viewer.py        # Componente Streamlit del visor
├── config.py              # Configuración centralizada
└── tests/
    └── test_trace_parser.py  # Tests del parser
```

### TraceModel (Estructura Interna)

El parser convierte cualquier formato de respuesta en un `TraceModel` normalizado:

```python
TraceModel {
    meta: TraceMeta           # Metadata (model, status, has_tools, has_reasoning)
    steps: list[TraceStep]    # Pasos ordenados del proceso
    final_outputs: list[FinalOutput]  # Outputs finales
    diagnostics: Diagnostics  # Errores, warnings, token usage
    raw: Any                  # JSON original (referencia)
}

TraceStep {
    id: str                   # ID único del paso
    index: int                # Índice en la secuencia
    kind: StepKind            # reasoning|tool_call|tool_result|output|error|unknown
    title: str                # Título para mostrar
    summary: str              # Resumen corto
    payload_raw_path: str     # JSONPath al dato original
    # ... campos específicos por tipo
}
```

### Extender el Parser

Para soportar nuevos tipos de nodos:

1. **Agregar nuevo `StepKind`** en `trace_parser.py`:
```python
class StepKind(Enum):
    # ... existing
    NEW_TYPE = "new_type"
```

2. **Agregar parsing** en la función correspondiente:
```python
def _parse_openai_responses_api(data, diagnostics):
    # ...
    elif item_type == "new_type":
        step = TraceStep(
            id=_generate_step_id(step_index, "new_type", item_id),
            index=step_index,
            kind=StepKind.NEW_TYPE,
            title="New Type",
            summary="...",
            # ... campos específicos
        )
        steps.append(step)
```

3. **Agregar renderizado** en `trace_viewer.py`:
```python
def _render_new_type_step(step: TraceStep):
    # Custom rendering logic
    pass

def _render_step(step, viewer_key):
    # ...
    elif step.kind == StepKind.NEW_TYPE:
        _render_new_type_step(step)
```

4. **Agregar tests** en `tests/test_trace_parser.py`

---

## Ejecutar Tests

```bash
# Instalar pytest si no está instalado
pip install pytest

# Ejecutar tests
pytest tests/ -v

# Ejecutar con cobertura
pip install pytest-cov
pytest tests/ --cov=. --cov-report=html
```

---

## Estructura de la Base de Datos

El dashboard lee de una base de datos DuckDB con las siguientes tablas:

- `runs`: Ejecuciones de experimentos
- `queries`: Queries individuales por run
- `artifacts`: Artefactos asociados (prompts, respuestas, outputs)

Los artefactos `response_bbdd`, `response_context` y `output` se muestran con el Trace Viewer.
