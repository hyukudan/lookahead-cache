# Strix Lookahead

**Componentes propios para acelerar la inferencia especulativa, diseñados primero
para Strix Halo y con backend CUDA independiente.**

[English and build commands](README.md)

La versión 0.3.0 conecta la cabeza de transiciones de rango 256 y su caché al motor
HIP mediante un parche privado y reversible. Añade una campaña que compara la
ruta original, la biblioteca sin caché y la biblioteca con caché. El motor conserva
la verificación, el estado del modelo y la corrección del muestreo.

## Qué contiene

| Componente | Estado |
| --- | --- |
| Caché genérica con API C, LRU y lecturas protegidas | Implementada y probada en CPU |
| Cabeza FP16/FP32 y caché directa de resultados | Implementadas y comparadas con la misma ruta sin caché |
| Backend HIP para `gfx1151` | Código compilado; ejecución en Strix pendiente |
| Backend CUDA para RTX | Código compilado; ejecución en GPU pendiente |
| Selector de candidatos con certificado y búsqueda completa de respaldo | Referencia CPU con pruebas exhaustivas y aleatorias |
| Adaptador privado para el motor HIP | Implementado; ejecutable y pruebas nativas compilados para `gfx1151` |
| Campaña de generación completa | Automatizada; ejecución con GPU y pesos reales pendiente |

Los dos backends mantienen dos kernels por paso. En un acierto leen el sesgo
almacenado y calculan el ganador; en un fallo reproducen la operación original y
guardan el resultado. No hay asignación de memoria, copia a CPU ni sincronización
del host dentro del encolado de un paso.

Cada instancia fija los pesos y el vocabulario ordenado, y pertenece a un único
stream. El crecimiento o cambio de ese vocabulario requiere sustituir la
instancia. La capacidad cero permite comparar la misma operación sin caché.

## Integración privada en 0.3.0

El adaptador asigna una caché independiente a cada solicitud y encadena los cinco
tokens propuestos dentro de la GPU. Conserva dos kernels por paso y fusiona la
publicación de resultados con el cálculo de confianza. Los estados y contadores
regresan junto a la transferencia que el motor ya realizaba.

La reserva de memoria incluye la caché de cada contexto. Un cambio real del
vocabulario reconstruye su binding e invalida las entradas anteriores; cambiar de
solicitud conserva su caché. La traza permite comprobar los aciertos registrados y
simular otras capacidades sin deducir una velocidad a partir del hit rate.

La campaña exige primero una prueba nativa de propuestas, confianza, embeddings
y logits con los pesos reales. Después comprueba los tokens finales, los hashes
de las filas conservadas y el estado final frente al motor original. Solo entonces
ejecuta las pasadas de tiempo, con trazas desactivadas y orden rotativo.

La primera ruta está acotada a un dispositivo HIP, generación greedy, ventanas
fijas y concurrencias 1, 2 y 4. El adaptador CUDA, los dos nodos y el controlador
adaptativo requieren integración o validación adicional. El código compilado y las
pruebas CPU no demuestran todavía una mejora de velocidad en los Strix físicos.

El motor de origen es público. El parche se conserva en este repositorio privado
y está fijado a un commit y a las huellas de sus cinco archivos. El instalador no
crea commits ni hace push. La [guía de integración](docs/engine-integration.md)
contiene los comandos para compilar, aplicar o revertir el parche y ejecutar la
campaña completa.

## Mejoras de la revisión 0.2.1

- Se rechazan también los modos de redondeo SSE modificados por separado del
  entorno x87. La regresión reproduce un caso que podía cambiar el token ganador.
- La referencia CPU omite 129 sumas redundantes por producto de rango 256 y evita
  convertir el embedding en un acierto si nadie solicita esa salida.
- La reducción de ganadores HIP/CUDA utiliza 128 bytes de memoria compartida y
  una barrera por bloque, frente a 4.096 bytes y nueve barreras. Las operaciones
  que calculan las puntuaciones conservan su orden original. La ganancia temporal
  en GPU sigue pendiente de medición física.
- El certificado acepta también empates demostrablemente favorables por ID global,
  sin cambiar el ganador ni añadir un recorrido del vocabulario durante la consulta.
- Los colectores validan el contenido de los benchmarks, conservan los fallos y
  registran huellas de los ejecutables. Una salida vacía, incompleta o inconsistente
  ya no puede presentarse como una calificación superada.

La [revisión técnica](docs/review-0.2.1.md) detalla los cambios, las mediciones y sus
límites. La interfaz C y la política de caché siguen siendo compatibles.

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
