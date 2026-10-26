# StrixLookahead

**Biblioteca independiente de optimizaciones reutilizables de lookahead para inferencia LLM.**

[English](README.md) · [Español](README.es.md) · Versión **0.4.1**

StrixLookahead contiene una caché host, un operador de pares de rango 256 para
CPU/HIP/CUDA, un selector certificado experimental en CPU y sus pruebas y
herramientas de medición. El objetivo principal es AMD Strix Halo; el backend
CUDA permite integrar el componente en RTX. La aplicación enlaza esta
biblioteca con su propio motor, que se encarga de cargar y ejecutar el modelo,
verificar las propuestas y aplicar el muestreo.

La optimización principal reutiliza un vector de sesgo FP32 que depende del
token predecesor, siempre que los pesos, la aritmética y el vocabulario ordenado
sean los mismos. Los logits base actuales se utilizan en cada paso. Un acierto
de caché evita repetir los productos escalares de rango 256; la mejora depende
de la reutilización, las colisiones, los cambios de vocabulario y el coste del
resto de la aplicación.

La [nota sobre el método](docs/method.md) explica el paper de referencia, su
relación con esta implementación, los cambios realizados y las líneas pendientes. Los
[resultados registrados](docs/results.md) separan las mediciones y sus límites
de esas propuestas.

## Componentes

| Componente | Función | API pública |
| --- | --- | --- |
| Caché host | Vectores FP32 calculados por el llamador, préstamos de lectura/escritura y desalojo LRU | [`cache.h`](include/strix_lookahead/cache.h) |
| Referencia CPU de pares | Pesos FP16, aritmética FP32 definida, selección voraz y caché de sesgos con asignación directa | [`pair.h`](include/strix_lookahead/pair.h) |
| Operadores HIP / CUDA | Ejecución asíncrona ligada a un stream, con capacidad cero como referencia sin caché | [`device.h`](include/strix_lookahead/device.h) |
| Selector experimental CPU | Certificado del candidato ganador mediante cotas conservadoras, con recorrido completo si no certifica | [`selection.h`](include/strix_lookahead/selection.h) |
| Ejemplos y herramientas | Uso de las API, benchmarks sintéticos, pruebas de corrección y análisis de trazas | [`examples`](examples), [`tools`](tools) |

La interfaz tiene ABI C y la implementación usa C++20. La caché host genérica y
la del operador de pares tienen políticas de desalojo diferentes. El selector
requiere preparar snapshots en O(V); que el certificado cueste O(K) no demuestra
por sí solo una aceleración de la inferencia completa.

## Compilar y probar en CPU

Clona este mismo repositorio con una cuenta que tenga acceso:

```sh
git clone git@github.com:hyukudan/strix-lookahead.git
cd strix-lookahead

cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
./build-cpu/sl_cache_example
./build-cpu/sl_pair_example
```

Se necesita un compilador C++20 y CMake 3.21 o posterior. Python 3.10+ habilita
las pruebas opcionales de herramientas y los colectores de evidencia. Los
ejemplos utilizan arrays sintéticos pequeños y no necesitan archivos de modelo.
La [guía de uso](docs/usage.md) explica la compilación HIP para `gfx1151`, CUDA,
la instalación y las opciones disponibles.

## Integrar la biblioteca

Instala la compilación elegida y enlaza su target CMake:

```sh
cmake --install build-cpu --prefix "$PWD/local-install"
```

```cmake
find_package(StrixLookahead 0.4.1 CONFIG REQUIRED COMPONENTS cache)
target_link_libraries(my_application PRIVATE StrixLookahead::cache)
```

Al configurar la aplicación, añade el prefijo instalado a `CMAKE_PREFIX_PATH`.
Una compilación con GPU también exporta `StrixLookahead::hip` o
`StrixLookahead::cuda`. Se puede incorporar el código mediante `add_subdirectory`.
La [guía de integración](docs/integration.md) detalla ambas opciones y las
responsabilidades sobre aritmética, vocabulario, invalidación, memoria, streams
y comprobación de resultados.

## Medir el componente

Tras compilar para CPU:

```sh
python3 tools/run_cpu_benchmark.py \
  --executable build-cpu/sl_pair_benchmark \
  --output results/cpu-pair.json --pin -- \
  --vocab 8192 --steps 24 --repeats 5 --capacity 16
```

Usa un archivo de salida nuevo en cada ejecución. `--pin` fija el proceso a una
CPU permitida en Linux; omítelo si la plataforma no permite esa afinidad. La
comparación utiliza datos sintéticos y el mismo operador CPU con capacidad cero
como referencia. No mide la inferencia de un modelo completo ni otro motor.

Para GPU, `tools/qualify_device.py` exige que las pruebas nativas pasen antes de
medir. La [guía de benchmarks](docs/benchmarks.md) explica los comandos, qué
incluyen los tiempos y cómo se registran los fallos. `qualify_engine.py` es una
herramienta opcional que exige una aplicación externa con una CLI y un contrato
de recibos específicos; ni esa aplicación ni su adaptador vienen incluidos.

## Validación y alcance

El [informe de validación](docs/validation.json) separa las pruebas host, la
compilación y la ejecución nativa en dispositivos. En esta versión pasaron las
siete pruebas CTest host, el ejemplo de pares y los consumidores del paquete.
Se completó la compilación HIP para `gfx1151` y CUDA para las arquitecturas
`86`, `89`, `120`. No se han ejecutado pruebas/benchmarks nativos de GPU ni
modelos para esta versión.

La equivalencia y el rendimiento con GPU/modelos deben validarse en el hardware
y la aplicación concretos; aquí no se afirma una aceleración de modelos ni
superioridad frente a otros sistemas. Revisa también las pruebas omitidas: una
GPU no disponible no cuenta como cualificación aprobada.

- [Compilación y ejemplos de API](docs/usage.md)
- [Integración y contrato de corrección](docs/integration.md)
- [Metodología de medición y herramientas opcionales](docs/benchmarks.md)
- [Licencia](LICENSE), [avisos](NOTICE), [procedencia](PROVENANCE.json)
