# Strix Lookahead

**Componentes propios para acelerar la inferencia especulativa, diseñados primero
para Strix Halo.**

[English](README.md)

El objetivo es reducir trabajo del borrador y mejorar sus propuestas.
El motor conserva la verificación y el estado del modelo. HIP sobre `gfx1151` es
el primer destino; CUDA para RTX compartirá el contrato de integración.

La base actual es una **caché de referencia en CPU** para vectores FP32 completos:
API C, almacenamiento acotado, expulsión LRU, reservas, publicación explícita y
lecturas protegidas mediante leases. El motor suministra los valores; la biblioteca
conserva sus bytes.

Los backends GPU, la integración con el motor y el corrector contextual están
planificados. Todavía no se atribuye ninguna aceleración medida a este proyecto.

Comprobaciones locales:

```sh
make test
make example
```

Para CMake, véase el [README principal](README.md).

Cada caché pertenece a unos pesos, una identidad numérica y un vocabulario
**ordenado**. Cambiar los IDs o su orden exige invalidarla. Todas las llamadas
requieren serialización externa, y una lectura permanece protegida hasta liberar
su lease. Los punteros actuales corresponden a memoria del host.

Documentación: [arquitectura](docs/architecture.md),
[integración](docs/integration.md), [plan](docs/roadmap.md) y
[validación](docs/validation.md).
