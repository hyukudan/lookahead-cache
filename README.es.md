# Strix Lookahead

**Componentes propios para acelerar la inferencia especulativa, diseñados primero
para Strix Halo y con backend CUDA independiente.**

[English and build commands](README.md)

La versión 0.2 implementa una cabeza de transiciones de rango 256, una caché de sus
vectores FP32 y un selector experimental con certificado matemático. El motor
conserva la verificación, el estado del modelo y la corrección del muestreo.

## Qué contiene

| Componente | Estado |
| --- | --- |
| Caché genérica con API C, LRU y lecturas protegidas | Implementada y probada en CPU |
| Cabeza FP16/FP32 y caché directa de resultados | Implementadas y comparadas con la misma ruta sin caché |
| Backend HIP para `gfx1151` | Código compilado; ejecución en Strix pendiente |
| Backend CUDA para RTX | Código compilado; ejecución en GPU pendiente |
| Selector de candidatos con certificado y búsqueda completa de respaldo | Referencia CPU con pruebas exhaustivas y aleatorias |
| Integración con el motor y comparación de generación completa | Contrato documentado; conexión y medición pendientes |

Los dos backends mantienen dos kernels por paso. En un acierto leen el sesgo
almacenado y calculan el ganador; en un fallo reproducen la operación original y
guardan el resultado. No hay asignación de memoria, copia a CPU ni sincronización
del host dentro del encolado de un paso.

Cada instancia fija los pesos y el vocabulario ordenado, y pertenece a un único
stream. El crecimiento o cambio de ese vocabulario requiere sustituir la
instancia. La capacidad cero permite comparar la misma operación sin caché.

## Pruebas reproducibles

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
./build-cpu/sl_pair_benchmark --vocab 129280 --steps 12 --repeats 3 --capacity 16
```

El banco CPU usa datos sintéticos y compara el mismo componente con y sin caché.
Conserva los fallos iniciales y prueba reutilización, tokens nuevos, carga mixta y
colisiones. Las cifras no representan tokens por segundo de un LLM ni rendimiento
medido en Strix Halo.

El [README principal](README.md) incluye la compilación HIP/CUDA y los ejecutables
para medir con eventos GPU y grafos. Si no hay hardware o controlador disponible,
devuelven código 77; esa omisión no se presenta como una prueba superada.

El selector certificado evita la búsqueda completa solo cuando demuestra que el
ganador no puede estar entre los candidatos excluidos. Su preparación tiene coste
O(V+K), que debe amortizarse o integrarse en el productor de logits antes de medir
una ventaja. Su ejecución GPU y un corrector contextual entrenado son trabajos
posteriores.

Documentación: [arquitectura](docs/architecture.md),
[integración](docs/integration.md), [benchmarks](docs/benchmarks.md),
[validación](docs/validation.md) y [plan](docs/roadmap.md).
