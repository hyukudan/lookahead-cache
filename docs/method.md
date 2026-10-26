# Método: reutilización exacta del sesgo de transición

LookaheadCache estudia e implementa optimizaciones del cálculo de propuestas
para inferencia especulativa. Este repositorio reúne el método, el código
reutilizable, las pruebas y nuestros datos. La ejecución del modelo y la
integración en un producto pertenecen al software que use la biblioteca.

## Referencias y alcance

El trabajo se sitúa en la línea de corrección de borradores paralelos. Estas son
las referencias primarias revisadas, con sus versiones fijadas:

- **DSpark**, Xin Cheng y colaboradores, julio de 2026,
  [arXiv:2607.05147v1](https://arxiv.org/html/2607.05147v1), sección 3.1,
  ecuaciones 4–5. Combina un borrador paralelo con una etapa secuencial ligera.
  Su variante Markov añade a los logits un sesgo de transición factorizado de
  bajo rango, dependiente del token anterior. El artículo también desarrolla
  una política de verificación guiada por confianza.
- **DCOCO**, Roy Eisenstadt y colaboradores, octubre de 2026,
  [arXiv:2610.12570v1](https://arxiv.org/html/2610.12570v1), sección 4.
  Conserva varios candidatos por posición, puntúa en paralelo las transiciones
  entre candidatos adyacentes y recorre después sus elecciones. Su corrector
  utiliza estados ocultos y logits del bloque actual: su puntuación completa
  no queda determinada solo por los IDs de dos tokens.

El [enlace inicial proporcionado](https://x.com/roy_eisenstadt/status/2114414151519121441)
y las [referencias BibTeX](../references.bib) permiten seguir la investigación.
Los enlaces a los manuscritos conservan la atribución de las ideas originales.
Las cifras publicadas por sus autores no son mediciones de LookaheadCache.

**La implementación actual optimiza el operador Markov estático de rango 256.**
No implementa el corrector contextual completo de DCOCO, su entrenamiento ni un
sistema de serving completo. Tampoco se presenta como un nuevo modelo entrenado.
La contribución de este repositorio es el código de ejecución y reutilización,
sus contratos de exactitud, sus herramientas y la evaluación que podamos
reproducir para nuestro caso de uso.

## Qué se calcula y qué se guarda

En la notación de nuestra [API de pares](../include/strix_lookahead/pair.h), `u`
es el ID global del token anterior y `g(j)` el ID global de la fila candidata
`j`. El operador implementa:

$$
b_{u,j}=\operatorname{dot256}_{\mathrm{contrato}}(H_{g(j)},E_u),
\qquad s_{t,j}=\operatorname{fl32}(a_{t,j}+b_{u,j}).
$$

`H` y `E` son las matrices inmutables del cabezal y de embeddings; `a` son los
logits base producidos para el paso actual. En el layout seleccionado, las filas
de `H` ya están en el orden `g` y se accede directamente a `H[j]`.

Mientras no cambien los pesos, el orden del vocabulario y el contrato numérico,
`b[u, :]` vuelve a ser el mismo vector cuando reaparece `u`. La biblioteca guarda
ese vector completo en FP32. En un acierto lee el vector, suma los **logits base
actuales** y vuelve a seleccionar el ganador. No reutiliza el ganador de un paso
anterior ni congela la parte contextual.

El [ejemplo ejecutable](../examples/pair_example.cpp) lo demuestra con datos
sintéticos: al reutilizar el mismo token anterior y modificar los logits base,
los ganadores cambian de 11 a 7 y a 3. La selección y cada valor del sesgo se
comparan bit a bit con una instancia de capacidad cero.

La identidad lógica de una entrada incluye pesos, aritmética, vocabulario
ordenado y token anterior. La implementación fija los tres primeros elementos
en cada instancia y usa `u` para buscar su entrada. Los identificadores opacos
de la API no calculan ni validan hashes de los archivos del modelo. Los arrays
prestados deben seguir inmutables y vivos; un cambio de pesos o mapping exige
crear una instancia nueva. `reset` solo invalida entradas.

## Optimizaciones que contiene el código

| Pieza | Implementación actual | Efecto que puede comprobarse |
| --- | --- | --- |
| Reutilización del sesgo | Vectores FP32 completos, caché con asignación directa de slots y capacidad configurable | Un acierto omite los productos del cabezal de pares |
| Referencia CPU | Conserva las 31 sumas de reducción que contribuyen al resultado observable | Evita calcular resultados intermedios descartados, manteniendo el orden de operandos |
| Lecturas GPU en aciertos | Hilos consecutivos leen filas consecutivas del vector guardado | El camino de acierto evita volver a leer la matriz del cabezal para calcular el sesgo |
| Reducción GPU | Subgrupos lógicos de 32 hilos y ocho resultados parciales por bloque | Usa 128 bytes de almacenamiento compartido explícito y una barrera por reducción de bloque |
| Trabajo asíncrono | Workspace fijo y dos lanzamientos por paso | `enqueue` no asigna memoria, sincroniza el host ni copia resultados al host |
| Caché host genérica | Vectores suministrados por el consumidor, LRU y leases de lectura/escritura | Permite reutilizar otros cálculos bajo un contrato de propiedad explícito |
| Selector certificado | Referencia CPU con cota del vocabulario excluido y fallback exacto | Puede certificar un ganador entre candidatos cuando la cota lo permite |

Estas propiedades se pueden inspeccionar en [pair.cpp](../src/pair.cpp),
[device_kernels.hpp](../src/device_kernels.hpp),
[device_runtime.hpp](../src/device_runtime.hpp), [cache.cpp](../src/cache.cpp)
y [selection.cpp](../src/selection.cpp). Los recuentos de operaciones, memoria
compartida y barreras describen el código; no sustituyen una medición en GPU.

El selector certificado es una API separada y experimental en CPU. Preparar sus
snapshots recorre el vocabulario completo y asigna memoria. Su consulta sobre
`K` candidatos solo resulta útil si se amortiza esa preparación. No está
conectado automáticamente al operador de pares ni a sus kernels GPU.

## Exactitud y condiciones de uso

El operador de pares usa entradas IEEE binary16, productos y sumas separados en
FP32 y un orden de reducción definido. No se habilita una contracción FMA que
cambie ese contrato. Los empates favorecen el menor ID global, los NaN se ignoran
en la selección y un paso sin ganador válido devuelve un estado de error. La
referencia CPU exige redondeo al más cercano y subnormales sin eliminación.

Un acierto debe entregar también el embedding del token anterior si el
consumidor lo pide para calcular confianza. La biblioteca no guarda ni sustituye
la confianza contextual del motor. Los buffers diagnósticos se rellenan tanto
en aciertos como en fallos.

Cada instancia GPU pertenece a un dispositivo y a un stream. El motor mantiene
el orden de productores y consumidores y comprueba el estado del resultado
cuando termina el trabajo. `SL_OK` al encolar solo confirma los lanzamientos.
La memoria compartida de una APU no elimina estas obligaciones de orden y vida
útil. Véase el [contrato de integración](integration.md).

La equivalencia de un componente tampoco demuestra por sí sola que una
integración preserve la generación completa. El software receptor sigue siendo
responsable de verificación, muestreo, estado KV/recurrente y commit/rollback.

## Memoria, localidad y punto de equilibrio

Con `C` entradas y `V` candidatos, el payload es `4*C*V` bytes; hay que añadir
metadata, workspace y costes del asignador. Para `V=129280`:

| Entradas | Payload por instancia |
| ---: | ---: |
| 4 | 1.97 MiB |
| 16 | 7.89 MiB |
| 64 | 31.56 MiB |
| 128 | 63.12 MiB |

La memoria se multiplica por las instancias residentes. Un presupuesto pequeño
puede provocar colisiones entre tokens que comparten slot. Ampliarlo puede
reducirlas, pero no garantiza beneficio: un mapping que se reconstruye con
frecuencia invalida entradas aunque haya espacio libre.

El coste de un acierto sigue incluyendo la lectura de `V` valores, su suma con
los logits actuales y la reducción del ganador. El coste de un fallo incorpora
el producto y la publicación del vector. Guardar valores poco reutilizados
puede añadir trabajo sin ahorrar cálculos posteriores.

Como modelo de decisión propio, si `h` es la fracción de aciertos, el tiempo
medio se puede estimar con:

$$
T_{\mathrm{cache}}\approx hT_{\mathrm{hit}}+(1-h)T_{\mathrm{miss}}
+\frac{T_{\mathrm{binding}}}{N}.
$$

Los tiempos deben medirse en el dispositivo y workload reales. Las
[trazas](benchmarks.md#optional-predecessor-trace-analysis) permiten estudiar reutilización,
colisiones y cambios de época. Una simulación de capacidad no mide tiempo de
ejecución y una traza incompleta no justifica una política de caché.

Aunque el operador se acelere mucho, el efecto final depende de cuánto tiempo
ocupaba. Si representase el 10% de una ejecución y se acelerase diez veces,
manteniendo idéntico el resto, el límite de ese cambio sería aproximadamente
`1/(0.9 + 0.1/10) = 1.10x`. Es un ejemplo matemático, no un resultado del motor.

## Datos actuales y preguntas abiertas

Nuestros [resultados CPU](results.md) conservan configuración, perfiles,
repeticiones, hashes y límites de lo medido. El [registro de validación](validation.json)
separa las pruebas host de la compilación HIP/CUDA. Todavía falta ejecutar los
contratos en Strix Halo y RTX físicos, y después medir generaciones reales con
el mismo modelo, prompts, política de verificación y límites de memoria.

Las siguientes líneas son propuestas de investigación:

1. Elegir capacidad o desactivar la caché según reutilización, colisiones y
   coste observado, incluyendo la reconstrucción del vocabulario.
2. Evaluar selección certificada en GPU solo si sus cotas pueden prepararse o
   reutilizarse con menos coste que el barrido que evita.
3. Medir concurrencia y tráfico de memoria por stream en el hardware objetivo;
   un acierto rápido aislado puede competir con el modelo por ancho de banda.
4. Para correctores contextuales, separar cualquier transformación estática
   reutilizable de los términos dependientes del bloque. Aplicar la clave de
   token anterior a una puntuación contextual completa daría reutilización
   incorrecta. Esa extensión requeriría otra API y validación propia.

La integración con un motor se revisa y mantiene en el repositorio de ese motor.
Este árbol conserva el método y el componente reutilizable para que el software
propio pueda adoptar, medir y adaptar las optimizaciones.
