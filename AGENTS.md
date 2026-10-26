# Repository scope

StrixLookahead is a standalone optimization library. Keep its source tree focused
on the library, public API, examples, tests, benchmarks and integration guidance.
Document the method with primary research references and preserve reproducible
project-owned measurements separately from performance reported in papers.
Do not import an inference engine, its application tree, model assets, kernel
drivers or historical benchmark archive into this repository.

External engines remain separate dependencies selected by the caller. Integration
examples and qualification tools must state the API or protocol they require;
they must not imply that an external engine is shipped or supported automatically.

Preserve the documented numerical, memory-lifetime and stream contracts. Keep
compiler validation, native-device correctness and full-model performance evidence
distinct. A successful build or synthetic benchmark does not establish a model
speedup. Preserve attribution for any adopted source.
